function [net, info] = SEGNET_BASIC(imdb, netF, inpt, varargin)
    
	% some common options
	trainer = @cnn_train_dag_seg_adv;

	opts.train.extractStatsFn = @extract_stats_segmentation2;
	opts.train.batchSize = 15;
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
	netE = varargin{1}{1};
    bopts.net = ovl_simplenn_move_extended(netE, 'gpu');
    
    	% ADAM optimizer
    	opts.train.alpha = 0.001;
    	opts.train.beta1 = 0.9;
    	opts.train.beta2 = 0.999;

	% organize data
	K = 2; % how many examples per domain	
	trainData1 = find((imdb.images.set == 1)  | (imdb.images.set == 2)); 
    trainData2 = find(imdb.images.set == 10);   
    trainData = cat(2, trainData1, trainData2(1:1000));
    valData = find(imdb.images.set == 3);
	% debuging code
    %trainData = trainData(1:10);
	%valData = valData(1:30);
	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];

     % defining batch policies 
     bopts_train.mode           = 'balanced';
     bopts_train.imdb           = imdb;
     bopts_train.balanceStruct  = [1, 4; 2, 6; 10, 3];
     bopts_train.maxNumBatches  = 150;
     bmanagerTrain              = BatchManagerBalanced(bopts_train);
     
     bopts_val.mode             = 'seq';
     bopts_val.domains          = [3];
     bopts_val.bsize            = 15;
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
	net.addLayer('conv1', dagnn.Conv('size', [7 7 3 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('bn1', dagnn.BatchNorm('numChannels', 64), {'conv1'}, {'bn1'}, {'bn1f', 'bn1b', 'bn1m'});
	net.addLayer('relu1', dagnn.ReLU(), {'bn1'}, {'relu1'}, {});
	net.addLayer('pool1', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu1'}, {'pool1', 'pool1_indices', 'sizes_pre_pool1', 'sizes_post_pool1'}, {});

	net.addLayer('conv2', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('bn2', dagnn.BatchNorm('numChannels', 64), {'conv2'}, {'bn2'}, {'bn2f', 'bn2b', 'bn2m'});
	net.addLayer('relu2', dagnn.ReLU(), {'bn2'}, {'relu2'}, {});
	net.addLayer('pool2', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu2'}, {'pool2', 'pool2_indices', 'sizes_pre_pool2', 'sizes_post_pool2'}, {});

	net.addLayer('conv3', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool2'}, {'conv3'},  {'conv3f'  'conv3b'});
	net.addLayer('bn3', dagnn.BatchNorm('numChannels', 64), {'conv3'}, {'bn3'}, {'bn3f', 'bn3b', 'bn3m'});
	net.addLayer('relu3', dagnn.ReLU(), {'bn3'}, {'relu3'}, {});
	net.addLayer('pool3', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu3'}, {'pool3', 'pool3_indices', 'sizes_pre_pool3', 'sizes_post_pool3'}, {});

	net.addLayer('conv4', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool3'}, {'conv4'},  {'conv4f'  'conv4b'});
	net.addLayer('bn4', dagnn.BatchNorm('numChannels', 64), {'conv4'}, {'bn4'}, {'bn4f', 'bn4b', 'bn4m'});
	net.addLayer('relu4', dagnn.ReLU(), {'bn4'}, {'relu4'}, {});
	net.addLayer('pool4', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu4'}, {'pool4', 'pool4_indices', 'sizes_pre_pool4', 'sizes_post_pool4'}, {});



	net.addLayer('unpool5', dagnn.Unpooling(), {'pool4', 'pool4_indices', 'sizes_pre_pool4', 'sizes_post_pool4'}, {'unpool5'}, {});
	net.addLayer('conv5', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool5'}, {'conv5'},  {'conv5f'  'conv5b'});
	net.addLayer('bn5', dagnn.BatchNorm('numChannels', 64), {'conv5'}, {'bn5'}, {'bn5f', 'bn5b', 'bn5m'});
	net.addLayer('relu5', dagnn.ReLU(), {'bn5'}, {'relu5'}, {});

	net.addLayer('unpool6', dagnn.Unpooling(), {'relu5', 'pool3_indices', 'sizes_pre_pool3', 'sizes_post_pool3'}, {'unpool6'}, {});
	net.addLayer('conv6', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool6'}, {'conv6'},  {'conv6f'  'conv6b'});
	net.addLayer('bn6', dagnn.BatchNorm('numChannels', 64), {'conv6'}, {'bn6'}, {'bn6f', 'bn6b', 'bn6m'});
	net.addLayer('relu6', dagnn.ReLU(), {'bn6'}, {'relu6'}, {});


	net.addLayer('unpool7', dagnn.Unpooling(), {'relu6', 'pool2_indices', 'sizes_pre_pool2', 'sizes_post_pool2'}, {'unpool7'}, {});
	net.addLayer('conv7', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool7'}, {'conv7'},  {'conv7f'  'conv7b'});
	net.addLayer('bn7', dagnn.BatchNorm('numChannels', 64), {'conv7'}, {'bn7'}, {'bn7f', 'bn7b', 'bn7m'});
	net.addLayer('relu7', dagnn.ReLU(), {'bn7'}, {'relu7'}, {});
    net.addLayer('drop', dagnn.DropOut('rate', 0.5), {'relu7'}, {'drop'}, {});

	net.addLayer('unpool8', dagnn.Unpooling(), {'drop', 'pool1_indices', 'sizes_pre_pool1', 'sizes_post_pool1'}, {'unpool8'}, {});
	net.addLayer('conv8', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool8'}, {'conv8'},  {'conv8f'  'conv8b'});
	net.addLayer('bn8', dagnn.BatchNorm('numChannels', 64), {'conv8'}, {'bn8'}, {'bn8f', 'bn8b', 'bn8m'});
	net.addLayer('relu8', dagnn.ReLU(), {'bn8'}, {'relu8'}, {});

	net.addLayer('classifier', dagnn.Conv('size', [1 1 64 11], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu8'}, {'classifier'},  {'classf'  'classb'});
	net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
	net.addLayer('objective', dagnn.LossWCE('weights', imdb.stats), {'prob','probx'}, 'objective');
	% -- end of the network

	% do the training!
	initNet(net, 1e-2*ones(1, 9), 1e-2*ones(1, 9));
	net.conserveMemory = false;

	info = trainer(net, imdb, @(i,b) getBatchProbs(bopts,i,b), opts.train);
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
