function processVideoSeqSP(infolder, outfolder, net_, opts)
	%opts.H = 180;
	%opts.W = 240;
	%opts.BSIZE = 1;

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
	gpuDevice(opts.GPU);

	% adjust net
	net = dagnn.DagNN.loadobj(net_);
	net.move('gpu');
	net.mode = 'test' ;

	% read data
	files = dir([infolder '/*.png']);
    	ref = imread([infolder '/' files(1).name]);
    	[H, W, CH] = size(ref);
    
	% data buffer
	BUFF = zeros(H, W, CH, opts.BSIZE, 'single');
    
    	figure;
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
    	    A = cell(size(norm_data, 4), 2000);


		for ll=1:size(norm_data, 4)
			im = im2single(norm_data(:,:,:,ll));
    			s = super_pixels(im);
        		visSP(im);
                pause(0.01);
        		for p=1:size(im,1)*size(im,2)
            			%s(p)+1
            			A{ll,s(p)+1}(end+1) = p; 
        		end
        	end	

            
           %figure; imshow(norm_data); 

            net.conserveMemory = true;
            net.eval({'input', gpuArray(norm_data./127.0), 'segms', A});
            net.conserveMemory = false;

            % obtain the CNN otuput
            scores = net.vars(net.getVarIndex(opts.output_point)).value;
            scores = squeeze(gather(scores)); 
            [outW, outL] = max(scores, [], 3);

            %Hmap = squeeze(sum(-scores.*log2(scores+eps), 3));
            %Vmap = squeeze(var(scores, 0, 3));
            
            %figure; imagesc(outW(:,:,1,1));
            %figure; imagesc(Vmap(:,:,1));
            %figure; imagesc(Hmap(:,:,1));
            
            %W = outW(:,:,:,1);
            %W2 = max(W - mean(W(:)), 0); W2 = W2 ./ max(W2(:));
            %mean(W2(W2>0))
            %figure; imagesc(W2);
            
            % save results
            for j=1:size(outL, 4)
                I = squeeze(outL(:,:,:,j));
                I2 = uint8(reshape(kittiColorMap(I+1, :, :), size(I, 1), size(I, 2), 3));
                RGB = uint8(BUFF(:,:,:,j));
                        LABS = imresize(I2, [H, W], 'nearest');

                O = LABS + 0.4*RGB;
                        O = cat(2, RGB, O);
                %imshow(O);
            %pause;

                % save prediction
                imwrite(O, [outfolder '/' files(i-opts.BSIZE+j).name]);			
            end
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
