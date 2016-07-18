function generateResults(imdb, net_, id, batchSize, outputFolder, opts)
	predictionFolder = [outputFolder '/Prediction'];
	GTFolder = [outputFolder '/GT'];
	visualizationFolder = [outputFolder '/Visualization'];
	sideFolder = [outputFolder '/Side'];
	if(~exist(predictionFolder, 'dir'))
		mkdir(predictionFolder);
	end
	if(~exist(visualizationFolder, 'dir'))
		mkdir(visualizationFolder);
	end
	if(~exist(GTFolder, 'dir'))
		mkdir(GTFolder);
	end
	if(~exist(sideFolder, 'dir'))
		mkdir(sideFolder);
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


	testIndices = find(imdb.images.set == id);
	RGBX = imdb.images.data(:, :, :, testIndices);
	GT = imdb.images.labels(:,:,1,testIndices);
	M = numel(testIndices);
    
    ii = 1;

     
    k = 1;
    for i=1:ceil(M/batchSize)
		idx1 = (i-1)*batchSize + 1;
		idx2 = min(idx1+batchSize-1, M);


		net.conserveMemory = true;

	    if(opts.divide)
            net.eval({'input', gpuArray(RGBX(:, :, :, idx1:idx2)./127.0)});
else
            net.eval({'input', gpuArray(RGBX(:, :, :, idx1:idx2))});
end
            net.conserveMemory = false;

            % obtain the CNN otuput
            scores = net.vars(net.getVarIndex('prob')).value;
            scores = squeeze(gather(scores)); 
            [outW, outL] = max(scores, [], 3);

		input2 = GT(:, :, 1, idx1:idx2);
		vInd =  idx1:idx2;
		for j=1:size(outL, 4)
			I = squeeze(outL(:,:,:,j)+1);
			I2 = uint8(reshape(kittiColorMap(I, :, :), size(I, 1), size(I, 2), 3));

			% save prediction
			imwrite(I2, [predictionFolder '/' num2str(ii, '%05d') '.png']);

			% save visualization
			RGB = uint8(squeeze(200*RGBX(:, :, :, vInd(j))/5.0));	
            		V = 0.2*RGB + 0.8*I2;
			imwrite(V, [visualizationFolder '/' num2str(ii,  '%05d') '.png']);



			%k = k + 1;
			I6 = [RGB; V];
			imwrite(I6, [sideFolder '/' num2str(ii, '%05d') '.png']);

			% save GT
			I3 = squeeze(input2(:,:,:,j));
			I4 = uint8(reshape(kittiColorMap(I3+1, :, :), size(I3, 1), size(I3, 2), 3));
			imwrite(I4, [GTFolder '/' num2str(ii,  '%05d') '.png']);
			
			ii = ii + 1;
		end
	end
		       
end

