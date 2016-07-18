function processVideoSeqUncertain(infolder, outfolder, net_, opts)
	%opts.H = 180;
	%opts.W = 240;
	%opts.BSIZE = 10;
	opts.insertDropAfterLayer = 'drop';
	opts.MAX_ITERS_DROP = 50;
    opts.l = 1;
    opts.rate = 0.65;

	if(~exist(outfolder, 'dir'))
		mkdir(outfolder);
	end	

	kittiColorMap = [0 0 0
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
                    0 128 192];

	% set GPU
	run(fullfile(fileparts(mfilename('fullpath')),'../../matlab/', 'vl_setupnn.m')) ;
	gpuDevice(1);

	% adjust net
	net = dagnn.DagNN.loadobj(net_);
	for l=1:numel(net.layers)
		for i=1:numel(net.layers(l).inputs)
			if(strcmp(net.layers(l).inputs{i}, opts.insertDropAfterLayer))
				net.layers(l).inputs{i} = 'dropX';
			end
		end
	end
	net.addLayer('dropX', dagnn.DropOutPermanent('rate', opts.rate), {opts.insertDropAfterLayer}, {'dropX'}, {});


	net.move('gpu');
	net.mode = 'test' ;

	% read data
	files = dir([infolder '/*.png']);
    	ref = imread([infolder '/' files(1).name]);
    	[H, W, CH] = size(ref);
    
	% data buffer
	BUFF = zeros(H, W, CH, opts.BSIZE, 'single');
    
    	%figure;
	for i=1:numel(files)
        ic = single(imread([infolder '/' files(i).name]));
        
        % image size has change
        if(numel(ic) ~= H*W*CH)
            ic = imresize(ic, [H,W]);
        end
        
        BUFF(:,:,:, mod(i-1, opts.BSIZE)+1) = ic;

        % buffer ready
        if(mod(i, opts.BSIZE) == 0)
            norm_data = normalize(BUFF, opts);

            list_scores = zeros(opts.MAX_ITERS_DROP, size(norm_data, 1), size(norm_data, 2), 11, opts.BSIZE);
            list_labs = zeros(opts.MAX_ITERS_DROP, size(norm_data, 1), size(norm_data, 2), 1, opts.BSIZE);

            for itersDrop=1:opts.MAX_ITERS_DROP
                    net.conserveMemory = true;
                    net.eval({'input', gpuArray(norm_data)});
                    net.conserveMemory = false;

                    scores = net.vars(net.getVarIndex('prob')).value;
                    scores = squeeze(gather(scores)); 
                    [outW, outL] = max(scores, [], 3);
                    
                    list_labs(itersDrop, :, :, :) = outW;
                    %list_scores(itersDrop, :, :, :, :) = scores;
            end

            varlab = squeeze(var(list_labs));
            figure; imagesc(varlab(:,:,1));
            
            %varscore = squeeze(var(list_scores)); %(((opts.l*opts.l*(1-0.5))/(2*opts.MAX_ITERS_DROP))^-1) + squeeze(var(list_scores));
            %varscore_i = varscore(:,:,:,1);
            %varscore_i = varscore_i ./ max(varscore_i(:));
            %for j=1:11
             %   figure; imagesc(varscore_i(:,:,j));
            %end
            pause;
	    
        end
    end 
end

function [out_data] = normalize(in_data, opts)
    out_data = imresize(in_data, [opts.H, opts.W]);
	out_data = vl_nnspnorm(out_data, [7,7, 1, 0.4]);

	H = size(out_data, 1);
	W = size(out_data, 2);
	N = size(out_data, 4);
	m = mean(reshape(permute(out_data, [1,2,4,3]), [H*W*N,3]));
	out_data = out_data - repmat(permute(m, [1,3,2]), [H, W, 1, N]);
	out_data = min(out_data, 3);
	out_data = max(out_data, -3);

 	range_min = min(out_data(:));
 	range_max = max(out_data(:));
 	range_multiplier = 127./max(abs(range_min),range_max);
 	out_data = out_data.*range_multiplier;
end

function [net_test] = adjust_net_for_test(net, opts)
    for i=numel(net.layers):-1:1
         if(~strcmp(net.layers{i}.name, opts.output_point_test))
            net.layers(i) = []; 
         else
             break;
         end
    end
    
    if(strcmp(net.layers{end}.type, 'softmaxloss') || strcmp(net.layers{end}.type, 'softmaxlossWeights') || strcmp(net.layers{end}.type, 'softmaxlossWeights2'))
       net.layers{end}.type = 'softmax'; 
       net.layers{end}.inputs = net.layers{end}.inputs{1};
    end
    
    net_test = net;
end
