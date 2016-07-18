function [res, dzdws] = ovl_dagnnBNFast(net, inputs, res, arcs, varargin)
% ovl_DAGNN  Evaluates a DAG CNN

	% some important options
	opts.cudaKernel1 = struct();
	opts.sync = false ;
	opts.disableDropout = false ;
	opts.forgetRelu = false;
	opts.conserveMemory = 0;
	opts.debugmem = false;
	opts.currentDomain = 1;
	opts.removeUntilLayer = -1;
	opts.compute_accum_bnorm = false;
	opts.fix_bnorm = false;
       opts.output_point_test = 'end';

	% option parser
	opts = ovl_argparse(opts, varargin);

	% let's create the dag structure (graph)
	if (nargin <= 4) || isempty(arcs)
  		arcs = ovl_dagnn_getarcs(net, inputs, false);
	end
	assert(numel(arcs.bufferNames) == numel(arcs.maps)+numel(inputs));
	numArcs = numel(arcs.maps);

	numLayers = numel(net.layers) ;
	numInputs = numel(inputs);
	inputs_x = {inputs.x};
	input_isgpu = cellfun(@(a) isa(a, 'gpuArray'), inputs_x);
	gpuMode = all(input_isgpu) ;
	if ~gpuMode && any(input_isgpu)
		error('Inputs mixed with CPU / GPU arrays'); 
	end

	% let's keep the intermediate results
	if (nargin <= 3) || isempty(res)
  	% Construct a new res structure
  		res = struct('x', [inputs_x cell(1,numLayers)], 'name', arcs.bufferNames, 'dzdx', cell(1,numLayers+numInputs), 'aux', cell(1,numLayers+numInputs), 'time', num2cell(zeros(1,numLayers+numInputs)), 'backwardTime', num2cell(zeros(1,numLayers+numInputs))) ;
	else
  		assert(numel(res) == numel(arcs.bufferNames));
  		% Fill in the inputs
  		[res(1:numel(inputs)).x] = deal(inputs_x{1:numel(inputs)}) ;
	end
    

	% forward loop
	for arci=1:numArcs
  		map = arcs.maps(arci);
  		l = net.layers{map.layerIdx};

  		inbis = map.inputIdxs; % indices of input buffers
  		outbi = map.outputIdx; % indices of an output buffer
  
		% which type of layer?
		switch l.type
    			case 'conv'
      				switch numel(inbis)
        				case 1
         					res(outbi).x = ovl_nnconv(res(inbis).x, l.weights{1}, l.weights{2}, 'pad', l.pad, 'stride', l.stride) ;  
					case 2
						res(outbi).x = ovl_nnconv(res(inbis(1)).x, res(inbis(2)).x, [], 'pad', l.pad, 'stride', l.stride) ;
        				case 3
						res(outbi).x = ovl_nnconv(res(inbis(1)).x, res(inbis(2)).x, res(inbis(3)).x, 'pad', l.pad, 'stride', l.stride) ;
        				otherwise
						error('Invalid use of ovl_nnconv. Too many inputs');
					end
      
			case 'combine'
				res(outbi).x = ovl_nncombine(res(inbis(1)).x, res(inbis(2)).x, l.weights{1});

			case 'frozenNet'
                l.params.dagnn_fcn = @ovl_dagnnBN;
				l.params.opts.cudaKernel1 = opts.cudaKernel1;
                		l.params.opts.conserveMemory = opts.conserveMemory;
				res(outbi).x = ovl_nnfrozennet(res(inbis).x, l.net, l.params);

			case 'doNothing'
				res(outbi).x = ovl_doNothing({res(inbis).x}, l.selector);
			case 'padder'
				res(outbi).x = ovl_nnpadder(res(inbis).x, l.padd);
			case 'convt'
				if(isfield(l, 'crop'))
                    %size(res(inbis).x)
                    %size(l.weights{1})
					res(outbi).x = ovl_nnconvt(res(inbis).x, l.weights{1}, l.weights{2}, 'Upsample', l.Upsample, 'Crop', l.crop) ;
				else
					res(outbi).x = ovl_nnconvt(res(inbis).x, l.weights{1}, l.weights{2}, 'Upsample', l.Upsample) ;
				end

			case 'sum'
				res(outbi).x = ovl_nnsum({res(inbis).x});

			case 'sub'
				res(outbi).x = ovl_nnsub({res(inbis).x});

			case 'times'
				switch numel(inbis)
					case 1
						res(outbi).x = ovl_nntimes(res(inbis).x, l.weights{1}) ;
					case 2
						res(outbi).x = ovl_nntimes(res(inbis(1)).x, res(inbis(2)).x) ;
					otherwise
						error('Invalid use of ovl_nntimes. Too many inputs');
					end
			case 'pool'
				assert(numel(inbis) == 1);
				res(outbi).x = ovl_nnpool(res(inbis).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
    
       			case 'poolInd2'
				kernel = opts.cudaKernel1;
				kernel.GridSize(1) = size(res(inbis).x, 3);
				kernel.GridSize(2) = size(res(inbis).x, 4);
				kernel.ThreadBlockSize = [512, 1, 1];
				[res(outbi).x, res(outbi).ind, res(outbi).size] = ovl_nnpoolInd2(res(inbis).x, l.stride, [], l.pad, kernel);
				%str0 = sprintf('Size before pooling [%d x %d] | after pooling [%d x %d]', size(res(inbis).x, 1), size(res(inbis).x, 2), size(res(outbi).x, 1), size(res(outbi).x, 2));
				%disp(str0);
    
      			case 'unpool'
      				ID = -1;
      				for lt=1:numel(net.layers)
        				if(strcmp(net.layers{lt}.name, l.ID))
            					ID = lt + numel(inputs);
            					break;
        				end
      				end
				[res(outbi).x] = ovl_nnunpool(res(inbis).x, res(ID).ind, res(ID).size);
                res(ID).ind = [];

			case 'normalize'
				assert(numel(inbis) == 1);
				res(outbi).x = ovl_nnnormalize(res(inbis).x, l.param) ;

			case 'spnorm'
				assert(numel(inbis) == 1)
				res(outbi).x = ovl_nnspnorm(res(inbis).x, l.param) ;

			case 'softmax'
      				res(outbi).x = ovl_nnsoftmax(res(inbis).x) ;

			case 'bsplit'
      				fieldName = sprintf('xD%d', l.train_domains(1));
				[res(outbi).x, inputs(2).(fieldName)] = ovl_nnbsplit(res(inbis).x, inputs(3).x, l.train_domains, inputs(2).x);
    
			case 'lploss'
				res(outbi).x = ovl_nnLp(res(inbis(1)).x, res(inbis(2)).x, l.p, []) ;
			case 'lossNoWeights'
				%assert(numel(inbis) == 1);
				res(outbi).x = ovl_nnlossNoWeights(res(inbis(1)).x, inputs(2).x) ;
			case 'loss'
				res(outbi).x = ovl_nnloss(res(inbis(1)).x, inputs(2).x) ;
			case 'crossEntropy'
				res(outbi).x = ovl_nnCrossEntropy(res(inbis(1)).x, res(inbis(2)).x) ;
			case 'crossEntropyWeights'
				res(outbi).x = ovl_nnCrossEntropyWeights(res(inbis(1)).x, res(inbis(2)).x, res(inbis(3)).x) ;	
			case 'lossE'
				res(outbi).x = ovl_nnlossE(res(inbis(1)).x, inputs(2).x) ;
			case 'lossSplit'
				res(outbi).x = ovl_nnloss(res(inbis(1)).x, inputs(2).xD1) ;
			case 'loss0'
     				res(outbi).x = ovl_nnloss0(res(inbis(1)).x, inputs(2).x) ;
			case 'lossDA'
				res(outbi).x = ovl_nnlossDA(res(inbis(1)).x, inputs(2).x, inputs(2).sets, l.alpha) ;
			case 'lossBackground'
				res(outbi).x = ovl_nnlossBackground(res(inbis(1)).x, inputs(2).x) ;
			case 'lossComposite'
				res(outbi).x = l.alpha*res(inbis(1)).x + l.beta*res(inbis(2)).x;
			case 'lossCompositeCond'
				if(opts.currentDomain == 1)
					res(outbi).x = l.alpha*res(inbis(1)).x;
				else
					res(outbi).x = l.beta*res(inbis(2)).x;
				end

			case 'softmaxloss'
				if(numel(inbis) == 1)
					if ~isfield(l, 'class')
						error('GT class for softmaxloss not found.'); 
					end;
					res(outbi).x = ovl_nnsoftmaxloss(res(inbis).x, l.class) ;
				else

					res(outbi).x = ovl_nnsoftmaxloss(res(inbis(1)).x, res(inbis(2)).x) ;
				end

			case 'softmaxlossWeights'
				res(outbi).x = ovl_nnsoftmaxlossWeights(res(inbis(1)).x, res(inbis(2)).x, l.W) ;

			case 'softmaxlossWeightsSplit'
				res(outbi).x = ovl_nnsoftmaxlossWeights(res(inbis(1)).x, inputs(2).xD2, l.W, l.classMap) ;
			case 'relu'
      				assert(numel(inbis) == 1);
				res(outbi).x = ovl_nnrelu(res(inbis).x) ;
			case 'noffset'
				assert(numel(inbis) == 1);
				res(outbi).x = ovl_nnnoffset(res(inbis).x, l.param) ;
			case 'dropout'
				assert(numel(inbis) == 1);
				if opts.disableDropout
					res(outbi).x = res(inbis).x ;
				else
					[res(outbi).x, res(outbi).aux] = ovl_nndropout(res(inbis).x, 'rate', l.rate);
				end
			case 'bnorm'
				if(opts.fix_bnorm)
					[res(outbi).x] = ovl_nnbnormSM(res(inbis).x, l.weights{1}, l.weights{2}, 3, net.layers{map.layerIdx}.mu, net.layers{map.layerIdx}.sigma);
				else
				   	[res(outbi).x] = ovl_nnbnormSM(res(inbis).x, l.weights{1}, l.weights{2}, 1, [], []);
				end

			case 'concatMultiFeat'
		      		res(outbi).x = ovl_nnconcatMultiFeat({res(inbis).x}, l.dim, l.numFeats) ;

			case 'custom'
				res(outbi) = l.forward(l, res(inbis), res(outbi)) ;
			otherwise
				error('Unknown layer type %s', l.type) ;
			end
  
			% THIS STAYS HERE FOR DEBUGGING
			%l
			%size(res(outbi).x)
  
            
         for ii=1:numel(inbis)
                if(~strcmp(res(inbis(ii)).name, opts.output_point_test))
                        arcs.bufCounters(inbis(ii)) = arcs.bufCounters(inbis(ii)) - 1;
                        res = clearBufs(res, arcs.bufCounters);
                end
        end

  	

	end

	
end


function a = superadd(a, b)
	% Addition with support for an empty array and cell arrays
	if isempty(a)
		a = b;
	else
		if iscell(a) && iscell(b)
			assert(numel(a) == numel(b));
			for i = 1:numel(a), a{i} = superadd(a{i}, b{i}); end
		else
			a = a + b;
        end
    end
end

function res = clearBufs(res, bufCounters)
	for bfi = find(bufCounters == 0)
	res(bfi).x = [];
    end
end

function debugmem(res, dzdw)
[cpum, gpum] = ovl_dagnn_buffersize(res, dzdw);
fprintf('CPU: % 8.2fMB \t GPU: % 8.2fMB\n', cpum./1024^2, gpum./1024^2);
end
