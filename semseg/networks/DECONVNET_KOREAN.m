function [net, info] = DECONVNET_KIREAN(imdb, netF, inpt, varargin)

	% some common options
	trainer = @cnn_train_dag_seg;

	opts.train.extractStatsFn = @extract_stats_segmentation;
	opts.train.batchSize = 64;
	opts.train.numEpochs = 20000;
	opts.train.continue = true ;
	opts.train.gpus = [1] ;
	opts.train.learningRate = [1e-2*ones(1, 10),  1e-2*ones(1, 5)];
	opts.train.weightDecay = 5e-4;
	opts.train.momentum = 0.9;
	opts.train.expDir = inpt.expDir;
	opts.train.savePlots = true;
	opts.train.numSubBatches = 1;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;

	% organize data
	K = 2; % how many examples per domain	
	trainData = find(imdb.images.set == 1);
	valData = find(imdb.images.set == 2);
	
	% debuging code
	trainData = trainData(1:15);
	valData = valData(1:15);

	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];

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
	net.addLayer('conv1_1', dagnn.Conv('size', [3 3 3 64], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'input'}, {'conv1_1'},  {'conv1_1f'  'conv1_1b'});
	net.addLayer('bn1_1', dagnn.BatchNorm('numChannels', 64), {'conv1_1'}, {'bn1_1'}, {'bn1_1f', 'bn1_1b', 'bn1_1m'});
	net.addLayer('relu1_1', dagnn.ReLU(), {'bn1_1'}, {'relu1_1'}, {});
	net.addLayer('conv1_2', dagnn.Conv('size', [3 3 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu1_1'}, {'conv1_2'},  {'conv1_2f'  'conv1_2b'});
	net.addLayer('bn1_2', dagnn.BatchNorm('numChannels', 64), {'conv1_2'}, {'bn1_2'}, {'bn1_2f', 'bn1_2b', 'bn1_2m'});
	net.addLayer('relu1_2', dagnn.ReLU(), {'bn1_2'}, {'relu1_2'}, {});
	net.addLayer('pool1', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu1_2'}, {'pool1', 'pool1_indices', 'sizes_pre_pool1', 'sizes_post_pool1'}, {});

	net.addLayer('conv2_1', dagnn.Conv('size', [3 3 64 128], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'pool1'}, {'conv2_1'},  {'conv2_1f'  'conv2_1b'});
	net.addLayer('bn2_1', dagnn.BatchNorm('numChannels', 128), {'conv2_1'}, {'bn2_1'}, {'bn2_1f', 'bn2_1b', 'bn2_1m'});
	net.addLayer('relu2_1', dagnn.ReLU(), {'bn2_1'}, {'relu2_1'}, {});
	net.addLayer('conv2_2', dagnn.Conv('size', [3 3 128 128], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu2_1'}, {'conv2_2'},  {'conv2_2f'  'conv2_2b'});
	net.addLayer('bn2_2', dagnn.BatchNorm('numChannels', 128), {'conv2_2'}, {'bn2_2'}, {'bn2_2f', 'bn2_2b', 'bn2_2m'});
	net.addLayer('relu2_2', dagnn.ReLU(), {'bn2_2'}, {'relu2_2'}, {});
	net.addLayer('pool2', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu2_2'}, {'pool2', 'pool2_indices', 'sizes_pre_pool2', 'sizes_post_pool2'}, {});

	net.addLayer('conv3_1', dagnn.Conv('size', [3 3 128 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'pool2'}, {'conv3_1'},  {'conv3_1f'  'conv3_1b'});
	net.addLayer('bn3_1', dagnn.BatchNorm('numChannels', 256), {'conv3_1'}, {'bn3_1'}, {'bn3_1f', 'bn3_1b', 'bn3_1m'});
	net.addLayer('relu3_1', dagnn.ReLU(), {'bn3_1'}, {'relu3_1'}, {});
	net.addLayer('conv3_2', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu3_2'}, {'conv3_2'},  {'conv3_2f'  'conv3_2b'});
	net.addLayer('bn3_2', dagnn.BatchNorm('numChannels', 256), {'conv3_2'}, {'bn3_2'}, {'bn3_2f', 'bn3_2b', 'bn3_2m'});
	net.addLayer('relu3_2', dagnn.ReLU(), {'bn3_2'}, {'relu3_2'}, {});
	net.addLayer('conv3_3', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu3_2'}, {'conv3_3'},  {'conv3_3f'  'conv3_3b'});
	net.addLayer('bn3_3', dagnn.BatchNorm('numChannels', 256), {'conv3_3'}, {'bn3_3'}, {'bn3_3f', 'bn3_3b', 'bn3_3m'});
	net.addLayer('relu3_3', dagnn.ReLU(), {'bn3_3'}, {'relu3_3'}, {});
	net.addLayer('pool3', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu3_3'}, {'pool3', 'pool3_indices', 'sizes_pre_pool3', 'sizes_post_pool3'}, {});

	net.addLayer('conv4_1', dagnn.Conv('size', [3 3 256 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'pool3'}, {'conv4_1'},  {'conv4_1f'  'conv4_1b'});
	net.addLayer('bn4_1', dagnn.BatchNorm('numChannels', 512), {'conv4_1'}, {'bn4_1'}, {'bn4_1f', 'bn4_1b', 'bn4_1m'});
	net.addLayer('relu4_1', dagnn.ReLU(), {'bn4_1'}, {'relu4_1'}, {});
	net.addLayer('conv4_2', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu4_2'}, {'conv4_2'},  {'conv4_2f'  'conv4_2b'});
	net.addLayer('bn4_2', dagnn.BatchNorm('numChannels', 512), {'conv4_2'}, {'bn4_2'}, {'bn4_2f', 'bn4_2b', 'bn4_2m'});
	net.addLayer('relu4_2', dagnn.ReLU(), {'bn4_2'}, {'relu4_2'}, {});
	net.addLayer('conv4_3', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu4_2'}, {'conv4_3'},  {'conv4_3f'  'conv4_3b'});
	net.addLayer('bn4_3', dagnn.BatchNorm('numChannels', 512), {'conv4_3'}, {'bn4_3'}, {'bn4_3f', 'bn4_3b', 'bn4_3m'});
	net.addLayer('relu4_3', dagnn.ReLU(), {'bn4_3'}, {'relu4_3'}, {});
	net.addLayer('pool4', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu4_3'}, {'pool4', 'pool4_indices', 'sizes_pre_pool4', 'sizes_post_pool4'}, {});

	net.addLayer('conv5_1', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'pool4'}, {'conv5_1'},  {'conv5_1f'  'conv5_1b'});
	net.addLayer('bn5_1', dagnn.BatchNorm('numChannels', 512), {'conv5_1'}, {'bn5_1'}, {'bn5_1f', 'bn5_1b', 'bn5_1m'});
	net.addLayer('relu5_1', dagnn.ReLU(), {'bn5_1'}, {'relu5_1'}, {});
	net.addLayer('conv5_2', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu5_2'}, {'conv5_2'},  {'conv5_2f'  'conv5_2b'});
	net.addLayer('bn5_2', dagnn.BatchNorm('numChannels', 512), {'conv5_2'}, {'bn5_2'}, {'bn5_2f', 'bn5_2b', 'bn5_2m'});
	net.addLayer('relu5_2', dagnn.ReLU(), {'bn5_2'}, {'relu5_2'}, {});
	net.addLayer('conv5_3', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu5_2'}, {'conv5_3'},  {'conv5_3f'  'conv5_3b'});
	net.addLayer('bn5_3', dagnn.BatchNorm('numChannels', 512), {'conv5_3'}, {'bn5_3'}, {'bn5_3f', 'bn5_3b', 'bn5_3m'});
	net.addLayer('relu5_3', dagnn.ReLU(), {'bn5_3'}, {'relu5_3'}, {});
	net.addLayer('pool5', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu5_3'}, {'pool5', 'pool5_indices', 'sizes_pre_pool5', 'sizes_post_pool5'}, {});

	net.addLayer('fc6', dagnn.Conv('size', [7 7 512 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool5'}, {'fc6'},  {'fc6f'  'fc6b'});
	net.addLayer('bn6', dagnn.BatchNorm('numChannels', 4096), {'fc6'}, {'bn6'}, {'bn6f', 'bn6b', 'bn6m'});
	net.addLayer('relu6', dagnn.ReLU(), {'bn6'}, {'relu6'}, {});

	net.addLayer('fc7', dagnn.Conv('size', [1 1 4096 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool6'}, {'fc7'},  {'fc7f'  'fc7b'});
	net.addLayer('bn7', dagnn.BatchNorm('numChannels', 4096), {'fc7'}, {'bn7'}, {'bn7f', 'bn7b', 'bn7m'});
	net.addLayer('relu7', dagnn.ReLU(), {'bn7'}, {'relu7'}, {});

	net.addLayer('decfc6', dagnn.ConvTranspose('size', [7 7 4096 512], 'hasBias', true, 'stride', [1, 1], 'crop', [0 0 0 0], 'upsample', [7,7]), {'relu7'}, {'decfc6'},  {'decfc6f'  'decfc6b'});
	net.addLayer('bndec', dagnn.BatchNorm('numChannels', 512), {'decfc6'}, {'bndec'}, {'bndecf', 'bndecb', 'bndecm'});
	net.addLayer('reludec', dagnn.ReLU(), {'bndec'}, {'reludec'}, {});

	net.addLayer('unpool5', dagnn.Unpooling(), {'reludec', 'pool5_indices', 'sizes_pre_pool5', 'sizes_post_pool5'}, {'unpool5'}, {});
	net.addLayer('decconv5_1', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'unpool5'}, {'conv5_1'},  {'conv5_1f'  'conv5_1b'});
	net.addLayer('bn5_1', dagnn.BatchNorm('numChannels', 512), {'conv5_1'}, {'bn5_1'}, {'bn5_1f', 'bn5_1b', 'bn5_1m'});
	net.addLayer('relu5_1', dagnn.ReLU(), {'bn5_1'}, {'relu5_1'}, {});
	net.addLayer('conv5_2', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu5_2'}, {'conv5_2'},  {'conv5_2f'  'conv5_2b'});
	net.addLayer('bn5_2', dagnn.BatchNorm('numChannels', 512), {'conv5_2'}, {'bn5_2'}, {'bn5_2f', 'bn5_2b', 'bn5_2m'});
	net.addLayer('relu5_2', dagnn.ReLU(), {'bn5_2'}, {'relu5_2'}, {});
	net.addLayer('conv5_3', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu5_2'}, {'conv5_3'},  {'conv5_3f'  'conv5_3b'});
	net.addLayer('bn5_3', dagnn.BatchNorm('numChannels', 512), {'conv5_3'}, {'bn5_3'}, {'bn5_3f', 'bn5_3b', 'bn5_3m'});
	net.addLayer('relu5_3', dagnn.ReLU(), {'bn5_3'}, {'relu5_3'}, {});
	net.addLayer('pool5', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu5_3'}, {'pool5', 'pool5_indices', 'sizes_pre_pool5', 'sizes_post_pool5'}, {});





	net.addLayer('conv2', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('bn2', dagnn.BatchNorm('numChannels', 64), {'conv2'}, {'bn2'}, {'bn2f', 'bn2b', 'bn2m'});
	net.addLayer('relu2', dagnn.ReLU(), {'bn2'}, {'relu2'}, {});
	net.addLayer('pool2', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu2'}, {'pool2', 'pool2_indices', 'sizes_pre_pool2', 'sizes_post_pool2'}, {});

	net.addLayer('unpool3', dagnn.Unpooling(), {'pool2', 'pool2_indices', 'sizes_pre_pool2', 'sizes_post_pool2'}, {'unpool3'}, {});
	net.addLayer('conv3', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool3'}, {'conv3'},  {'conv3f'  'conv3b'});
	net.addLayer('bn3', dagnn.BatchNorm('numChannels', 64), {'conv3'}, {'bn3'}, {'bn3f', 'bn3b', 'bn3m'});
	net.addLayer('relu3', dagnn.ReLU(), {'bn3'}, {'relu3'}, {});


	net.addLayer('unpool4', dagnn.Unpooling(), {'relu3', 'pool1_indices', 'sizes_pre_pool1', 'sizes_post_pool1'}, {'unpool4'}, {});
	net.addLayer('conv4', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool4'}, {'conv4'},  {'conv4f'  'conv4b'});
	net.addLayer('bn4', dagnn.BatchNorm('numChannels', 64), {'conv4'}, {'bn4'}, {'bn4f', 'bn4b', 'bn4m'});
	net.addLayer('relu4', dagnn.ReLU(), {'bn4'}, {'relu4'}, {});

	net.addLayer('classifier', dagnn.Conv('size', [1 1 64 11], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu4'}, {'classifier'},  {'classf'  'classb'});
	net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
	net.addLayer('objective', dagnn.LossSemantic('weights', true), {'prob','label'}, 'objective');
	% -- end of the network

	% do the training!
	initNet(net, 1e-2*ones(1, 5), 1e-2*ones(1, 5));
	net.conserveMemory = false;

	info = trainer(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'train', trainData, 'val', valData) ;
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

function initNet(net, F, B)
	net.initParams();
	%
	i = 1;
	for l=1:length(net.layers)
		% is a convolution layer?
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			net.params(f_ind).value = F(i)*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;

			net.params(b_ind).value = B(i)*randn(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 0.5;
			net.params(b_ind).weightDecay = 1;
			i = i + 1;
		end
	end
end
