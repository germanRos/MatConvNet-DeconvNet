function [net, info] = SEGNET_BASIC(imdb, netF, inpt, varargin)
    
	% some common options
	trainer = @cnn_train_dag_seg_adv;

	opts.train.extractStatsFn = @extract_stats_segmentation;
	opts.train.batchSize = 3;
	opts.train.numEpochs = 400;
	opts.train.continue = true ;
	opts.train.gpus = [1] ;
	opts.train.learningRate = [1e-8*ones(1, 10),  1e-2*ones(1, 5)];
	opts.train.weightDecay = 3e-4;
	opts.train.momentum = 0.9;
	opts.train.expDir = inpt.expDir;
	opts.train.savePlots = true;
	opts.train.numSubBatches = 1;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;
    
    	% ADAM optimizer
    	opts.train.alpha = 0.001;
    	opts.train.beta1 = 0.9;
    	opts.train.beta2 = 0.999;

	% organize data
	K = 2; % how many examples per domain	
	trainData1 = find((imdb.images.set == 1) ); 
    trainData2 = find(imdb.images.set == 1);   
    trainData = cat(2, trainData1, trainData2(1:5));
    valData = find(imdb.images.set == 3);
	% debuging code
    trainData = trainData(1:10);
valData = valData(1:30);
	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];

     % defining batch policies 
     bopts_train.mode           = 'balanced';
     bopts_train.imdb           = imdb;
     bopts_train.balanceStruct  = [1, 3];
     bopts_train.maxNumBatches  = 5;
     bmanagerTrain              = BatchManagerBalanced(bopts_train);
     
     bopts_val.mode             = 'seq';
     bopts_val.domains          = [3];
     bopts_val.bsize            = 3;
     bopts_val.imdb             = imdb;
     bmanagerVal                = BatchManagerSeq(bopts_val);
     
     opts.train.bmanagerTrain = bmanagerTrain;
     opts.train.bmanagerVal   = bmanagerVal;
   
	opts.train.classesNames = {'sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation', 'pole', 'car', 'sign', 'pedestrian', 'cyclist'};
	colorMap  = (1/255)*[		    
					    128 128 128
					    128 0 0
					    128 64 128
					    0 0 192
					    64 64 128
					    128 128 0
					    192 192 128
					    64 0 128
					    192 128 128
					    64 64 0
					    0 128 192
					    ];
	opts.train.colorMapGT = [0 0 0; colorMap];
	opts.train.colorMapEst = colorMap;

	% network definition
	net = dagnn.DagNN() ;
	net.addLayer('l1', dagnn.FrozenNet('pathmodel', '/DATA/Results/NewNetworks/FCN_D.mat', 'output_point', 'upsample'), {'input'}, {'l1'});
	
	net.addLayer('prob', dagnn.SoftMax(), {'l1'}, {'prob'}, {});
	net.addLayer('objective', dagnn.LossSemantic('weights', true), {'prob','label'}, 'objective');
	% -- end of the network

	% do the training!
	initNet(net, 1e-2*ones(1, 9), 1e-2*ones(1, 9));
	net.conserveMemory = false;

	info = trainer(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train);
end


% function on charge of creating a batch of images + labels
function inputs = getBatch(opts, imdb, batch)
	images = imdb.images.data(:,:,:,batch) ;
	labels = imdb.images.labels(:, :, :, batch) ;
	if opts.useGpu > 0
  		images = gpuArray(images);
		labels = gpuArray(labels);
	end

	inputs = {'input', images, 'label', labels} ;
end

function inputs = getBatchProbs(opts, imdb, batch)
	im = imdb.images.data(:,:,:,batch) ;
	labels = imdb.images.labels(:, :, :, batch);
	input = struct('name', {'data', 'label'}, 'x', {im, labels});
    for fi = 1:numel(input), input(fi).x = gpuArray(input(fi).x);  end

    opts2.forgetRelu = true;
   	opts2.conserveMemory = true;
	opts2.cudaKernel1 =  parallel.gpu.CUDAKernel('kernel_pooling.ptx','kernel_pooling.cu','my_poolingIndices');     
    res1 = ovl_dagnnBNFast(opts.net, input, [], [], opts2);

	res = res1(end).x;
		
    inputs = {'input', input(1).x, 'probx', res} ;
        
end


function initNet(net, F, B)
	net.initParams();
	%
	i = 1;
	for l=1:length(net.layers)
		% is a convolution layer?
		if(isa(net.layers(l).block, 'dagnn.Conv') || isa(net.layers(l).block, 'dagnn.ConvTranspose'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			net.params(f_ind).value = F(i)*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;
            net.params(f_ind).trainMethod = 'adam';
            net.params(f_ind).m = zeros(size(net.params(f_ind).value), 'single');
            net.params(f_ind).v = zeros(size(net.params(f_ind).value), 'single');
            net.params(f_ind).t = 1;


			net.params(b_ind).value = B(i)*randn(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 0.5;
			net.params(b_ind).weightDecay = 1;
            net.params(b_ind).trainMethod = 'adam';
            net.params(b_ind).m = zeros(size(net.params(b_ind).value), 'single');
            net.params(b_ind).v = zeros(size(net.params(b_ind).value), 'single');
            net.params(b_ind).t = 1;


			i = i + 1;
		end
	end
end
