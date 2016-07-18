function [net, info] = FCN_MULTIRES2(imdb, netF, inpt, varargin)

	% some common options
	trainer = @cnn_train_dag_seg_adv;

	opts.train.extractStatsFn = @extract_stats_segmentation;
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
    
    	% ADAM optimizer
    	opts.train.alpha = 0.001;
    	opts.train.beta1 = 0.9;
    	opts.train.beta2 = 0.999;

	% organize data
	K = 2; % how many examples per domain	
	trainData = find(imdb.images.set == 1);
	valData = find(imdb.images.set == 3);
	
	% debuging code
	%trainData = trainData(1:300);
	%valData = valData(1:300);

	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];

% defining batch policies 
     bopts_train.mode           = 'balanced';
     bopts_train.imdb           = imdb;
     bopts_train.balanceStruct  = [1, 7];
     bopts_train.maxNumBatches  = 150;
     bmanagerTrain              = BatchManagerBalanced(bopts_train);
     
     bopts_val.mode             = 'seq';
     bopts_val.domains          = [3];
     bopts_val.bsize            = 10;
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

	CLASSES = numel(opts.train.classesNames);

	% network definition
	net = dagnn.DagNN() ;

	% input pooling
	net.addLayer('pool_r2', dagnn.Pooling('method', 'max', 'poolSize', [4, 4], 'stride', [4 4], 'pad', [1 2 1 2]), {'input'}, {'input_r2'}, {});
	
	% BLOCK 1 res1 (r1)
	net.addLayer('conv1_1_r1', dagnn.Conv('size', [3 3 3 64], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'input'}, {'conv1_1_r1'},  {'conv1_1f'  'conv1_1b'});
	net.addLayer('relu1_1_r1', dagnn.ReLU(), {'conv1_1_r1'}, {'conv1_1x_r1'}, {});
	net.addLayer('conv1_2_r1', dagnn.Conv('size', [3 3 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv1_1x_r1'}, {'conv1_2_r1'},  {'conv1_2f'  'conv1_2b'});
	net.addLayer('relu1_2_r1', dagnn.ReLU(), {'conv1_2_r1'}, {'conv1_2x_r1'}, {});
	net.addLayer('pool1_r1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv1_2x_r1'}, {'pool1_r1'}, {});
	% BLOCK 1 res2 (r2)
	net.addLayer('conv1_1_r2', dagnn.Conv('size', [3 3 3 64], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'input_r2'}, {'conv1_1_r2'},  {'conv1_1f'  'conv1_1b'});
	net.addLayer('relu1_1_r2', dagnn.ReLU(), {'conv1_1_r2'}, {'conv1_1x_r2'}, {});
	net.addLayer('conv1_2_r2', dagnn.Conv('size', [3 3 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv1_1x_r2'}, {'conv1_2_r2'},  {'conv1_2f'  'conv1_2b'});
	net.addLayer('relu1_2_r2', dagnn.ReLU(), {'conv1_2_r2'}, {'conv1_2x_r2'}, {});
	net.addLayer('pool1_r2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv1_2x_r2'}, {'pool1_r2'}, {});
	% --- outputs: pool1_r1, pool1_r2

	% BLOCK 2 res1 (r1)
	net.addLayer('conv2_1_r1', dagnn.Conv('size', [3 3 64 128], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool1_r1'}, {'conv2_1_r1'},  {'conv2_1f'  'conv2_1b'});
	net.addLayer('relu2_1_r1', dagnn.ReLU(), {'conv2_1_r1'}, {'conv2_1x_r1'}, {});
	net.addLayer('conv2_2_r1', dagnn.Conv('size', [3 3 128 128], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv2_1x_r1'}, {'conv2_2_r1'},  {'conv2_2f'  'conv2_2b'});
	net.addLayer('relu2_2_r1', dagnn.ReLU(), {'conv2_2_r1'}, {'conv2_2x_r1'}, {});
	net.addLayer('pool2_r1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv2_2x_r1'}, {'pool2_r1'}, {});
	% BLOCK 2 res2 (r2)
	net.addLayer('conv2_1_r2', dagnn.Conv('size', [3 3 64 128], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool1_r2'}, {'conv2_1_r2'},  {'conv2_1f'  'conv2_1b'});
	net.addLayer('relu2_1_r2', dagnn.ReLU(), {'conv2_1_r2'}, {'conv2_1x_r2'}, {});
	net.addLayer('conv2_2_r2', dagnn.Conv('size', [3 3 128 128], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv2_1x_r2'}, {'conv2_2_r2'},  {'conv2_2f'  'conv2_2b'});
	net.addLayer('relu2_2_r2', dagnn.ReLU(), {'conv2_2_r2'}, {'conv2_2x_r2'}, {});
	net.addLayer('pool2_r2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv2_2x_r2'}, {'pool2_r2'}, {});
	% --- outputs: pool2_r1, pool2_r2

	% BLOCK 3 res1 (r1)
	net.addLayer('conv3_1_r1', dagnn.Conv('size', [3 3 128 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool2_r1'}, {'conv3_1_r1'},  {'conv3_1f'  'conv3_1b'});
	net.addLayer('relu3_1_r1', dagnn.ReLU(), {'conv3_1_r1'}, {'conv3_1x_r1'}, {});
	net.addLayer('conv3_2_r1', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv3_1x_r1'}, {'conv3_2_r1'},  {'conv3_2f'  'conv3_2b'});
	net.addLayer('relu3_2_r1', dagnn.ReLU(), {'conv3_2_r1'}, {'conv3_2x_r1'}, {});
	net.addLayer('conv3_3_r1', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv3_2x_r1'}, {'conv3_3_r1'},  {'conv3_3f'  'conv3_3b'});
	net.addLayer('relu3_3_r1', dagnn.ReLU(), {'conv3_3_r1'}, {'conv3_3x_r1'}, {});
	net.addLayer('pool3_r1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv3_3x_r1'}, {'pool3_r1'}, {});
	% BLOCK 3 res2 (r2)
	net.addLayer('conv3_1_r2', dagnn.Conv('size', [3 3 128 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool2_r2'}, {'conv3_1_r2'},  {'conv3_1f'  'conv3_1b'});
	net.addLayer('relu3_1_r2', dagnn.ReLU(), {'conv3_1_r2'}, {'conv3_1x_r2'}, {});
	net.addLayer('conv3_2_r2', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv3_1x_r2'}, {'conv3_2_r2'},  {'conv3_2f'  'conv3_2b'});
	net.addLayer('relu3_2_r2', dagnn.ReLU(), {'conv3_2_r2'}, {'conv3_2x_r2'}, {});
	net.addLayer('conv3_3_r2', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv3_2x_r2'}, {'conv3_3_r2'},  {'conv3_3f'  'conv3_3b'});
	net.addLayer('relu3_3_r2', dagnn.ReLU(), {'conv3_3_r2'}, {'conv3_3x_r2'}, {});
	net.addLayer('pool3_r2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv3_3x_r2'}, {'pool3_r2'}, {});
	% --- outputs: pool3_r1, pool3_r2

	% BLOCK 4 res1 (r1)
	net.addLayer('conv4_1_r1', dagnn.Conv('size', [3 3 256 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool3_r1'}, {'conv4_1_r1'},  {'conv4_1f'  'conv4_1b'});
	net.addLayer('relu4_1_r1', dagnn.ReLU(), {'conv4_1_r1'}, {'conv4_1x_r1'}, {});
	net.addLayer('conv4_2_r1', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv4_1x_r1'}, {'conv4_2_r1'},  {'conv4_2f'  'conv4_2b'});
	net.addLayer('relu4_2_r1', dagnn.ReLU(), {'conv4_2_r1'}, {'conv4_2x_r1'}, {});
	net.addLayer('conv4_3_r1', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv4_2x_r1'}, {'conv4_3_r1'},  {'conv4_3f'  'conv4_3b'});
	net.addLayer('relu4_3_r1', dagnn.ReLU(), {'conv4_3_r1'}, {'conv4_3x_r1'}, {});
	net.addLayer('pool4_r1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv4_3x_r1'}, {'pool4_r1'}, {});
	% BLOCK 4 res2 (r2)
	net.addLayer('conv4_1_r2', dagnn.Conv('size', [3 3 256 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool3_r2'}, {'conv4_1_r2'},  {'conv4_1f'  'conv4_1b'});
	net.addLayer('relu4_1_r2', dagnn.ReLU(), {'conv4_1_r2'}, {'conv4_1x_r2'}, {});
	net.addLayer('conv4_2_r2', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv4_1x_r2'}, {'conv4_2_r2'},  {'conv4_2f'  'conv4_2b'});
	net.addLayer('relu4_2_r2', dagnn.ReLU(), {'conv4_2_r2'}, {'conv4_2x_r2'}, {});
	net.addLayer('conv4_3_r2', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv4_2x_r2'}, {'conv4_3_r2'},  {'conv4_3f'  'conv4_3b'});
	net.addLayer('relu4_3_r2', dagnn.ReLU(), {'conv4_3_r2'}, {'conv4_3x_r2'}, {});
	net.addLayer('pool4_r2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv4_3x_r2'}, {'pool4_r2'}, {});
	% --- outputs: pool4_r1, pool4_r2
	
	% BLOCK 5 res1 (r1)
	net.addLayer('conv5_1_r1', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool4_r1'}, {'conv5_1_r1'},  {'conv5_1f'  'conv5_1b'});
	net.addLayer('relu5_1_r1', dagnn.ReLU(), {'conv5_1_r1'}, {'conv5_1x_r1'}, {});
	net.addLayer('conv5_2_r1', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv5_1x_r1'}, {'conv5_2_r1'},  {'conv5_2f'  'conv5_2b'});
	net.addLayer('relu5_2_r1', dagnn.ReLU(), {'conv5_2_r1'}, {'conv5_2x_r1'}, {});
	net.addLayer('conv5_3_r1', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv5_2x_r1'}, {'conv5_3_r1'},  {'conv5_3f'  'conv5_3b'});
	net.addLayer('relu5_3_r1', dagnn.ReLU(), {'conv5_3_r1'}, {'conv5_3x_r1'}, {});
	net.addLayer('pool5_r1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv5_3x_r1'}, {'pool5_r1'}, {});
	% BLOCK 5 res2 (r2)
	net.addLayer('conv5_1_r2', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool4_r2'}, {'conv5_1_r2'},  {'conv5_1f'  'conv5_1b'});
	net.addLayer('relu5_1_r2', dagnn.ReLU(), {'conv5_1_r2'}, {'conv5_1x_r2'}, {});
	net.addLayer('conv5_2_r2', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv5_1x_r2'}, {'conv5_2_r2'},  {'conv5_2f'  'conv5_2b'});
	net.addLayer('relu5_2_r2', dagnn.ReLU(), {'conv5_2_r2'}, {'conv5_2x_r2'}, {});
	net.addLayer('conv5_3_r2', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv5_2x_r2'}, {'conv5_3_r2'},  {'conv5_3f'  'conv5_3b'});
	net.addLayer('relu5_3_r2', dagnn.ReLU(), {'conv5_3_r2'}, {'conv5_3x_r2'}, {});
	net.addLayer('pool5_r2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv5_3x_r2'}, {'pool5_r2'}, {});
	% --- outputs: pool5_r1, pool5_r2

	% BLOCK 6 res1 (r1)
	net.addLayer('fc6_r1', dagnn.Conv('size', [7 7 512 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool5_r1'}, {'fc6_r1'},  {'fc6f'  'fc6b'});
	net.addLayer('relu6_r1', dagnn.ReLU(), {'fc6_r1'}, {'fc6x_r1'}, {});
	net.addLayer('dropout1_r1', dagnn.DropOut('rate', 0.5), {'fc6x_r1'}, {'fc6x2_r1'}, {});
	% BLOCK 6 res1 (r2)
	net.addLayer('fc6_r2', dagnn.Conv('size', [7 7 512 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool5_r2'}, {'fc6_r2'},  {'fc6f'  'fc6b'});
	net.addLayer('relu6_r2', dagnn.ReLU(), {'fc6_r2'}, {'fc6x_r2'}, {});
	net.addLayer('dropout1_r2', dagnn.DropOut('rate', 0.5), {'fc6x_r2'}, {'fc6x2_r2'}, {});
	% --- outputs: fc6x2_r1, fc6x2_r2

	
	% BLOCK 7 res1 (r1)
	net.addLayer('fc7_r1', dagnn.Conv('size', [1 1 4096 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'fc6x2_r1'}, {'fc7_r1'},  {'fc7f'  'fc7b'});
	net.addLayer('relu7_r1', dagnn.ReLU(), {'fc7_r1'}, {'fc7x_r1'}, {});
	net.addLayer('dropout2_r1', dagnn.DropOut('rate', 0.5), {'fc7x_r1'}, {'fc7x2_r1'}, {});
	net.addLayer('score_fr_r1', dagnn.Conv('size', [1 1 4096 CLASSES], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'fc7x2_r1'}, {'score_r1'},  {'score_frf'  'score_frb'});
	% BLOCK 7 res2 (r2)
	net.addLayer('fc7_r2', dagnn.Conv('size', [1 1 4096 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'fc6x2_r2'}, {'fc7_r2'},  {'fc7f'  'fc7b'});
	net.addLayer('relu7_r2', dagnn.ReLU(), {'fc7_r2'}, {'fc7x_r2'}, {});
	net.addLayer('dropout2_r2', dagnn.DropOut('rate', 0.5), {'fc7x_r2'}, {'fc7x2_r2'}, {});
	net.addLayer('score_fr_r2', dagnn.Conv('size', [1 1 4096 CLASSES], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'fc7x2_r2'}, {'score_r2'},  {'score_frf'  'score_frb'});
	% --- outputs: score_r1, score_r2

	% BLOCK 8 (r1)
	net.addLayer('score2_r1', dagnn.ConvTranspose('size', [4 4 CLASSES CLASSES], 'upsample', [2 2], 'hasBias', true, 'crop', [1 1 1 2], 'numGroups', 1), {'score_r1'}, {'score2_r1'},  {'score2f', 'score2b'});
	net.addLayer('score_pool4_r1', dagnn.Conv('size', [1 1 512 CLASSES], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool4_r1'}, {'score_pool4_r1'},  {'score_pool4f'  'score_pool4b'});
	net.addLayer('fuse_r1', dagnn.Sum(), {'score2_r1', 'score_pool4_r1'}, {'score_fused_r1'}, {});
	% BLOCK 8 (r2)
	net.addLayer('score2_r2', dagnn.ConvTranspose('size', [4 4 CLASSES CLASSES], 'upsample', [2 2], 'hasBias', true, 'crop', [2 1 1 1], 'numGroups', 1), {'score_r2'}, {'score2_r2'},  {'score2f', 'score2b'});
	net.addLayer('score_pool4_r2', dagnn.Conv('size', [1 1 512 CLASSES], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool4_r2'}, {'score_pool4_r2'},  {'score_pool4f'  'score_pool4b'});
	net.addLayer('fuse_r2', dagnn.Sum(), {'score2_r2', 'score_pool4_r2'}, {'score_fused_r2'}, {});
	% --- outputs: score_fused_r1, score_fused_r2

	% BLOCK 9 (r1)
	net.addLayer('score4_r1', dagnn.ConvTranspose('size', [4 4 CLASSES CLASSES], 'upsample', [2 2], 'hasBias', true, 'crop', [2 1 1 1], 'numGroups', 1), {'score_fused_r1'}, {'score4_r1'},  {'score4f', 'score4b'});
	net.addLayer('score_pool3_r1', dagnn.Conv('size', [1 1 256 CLASSES], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool3_r1'}, {'score_pool3_r1'},  {'score_pool3f'  'score_pool3b'});
	net.addLayer('fusex_r1', dagnn.Sum(), {'score4_r1', 'score_pool3_r1'}, {'score_final_r1'}, {});
	net.addLayer('upsample_r1', dagnn.ConvTranspose('size', [8 8 CLASSES CLASSES], 'upsample', [8 8], 'hasBias', true, 'crop', [2 2 0 0]), {'score_final_r1'}, {'bigscore_r1'},  {'upsamplef', 'upsampleb'});
	% BLOCK 9 (r2)
	net.addLayer('score4_r2', dagnn.ConvTranspose('size', [4 4 CLASSES CLASSES], 'upsample', [2 2], 'hasBias', true, 'crop', [1 1 1 1], 'numGroups', 1), {'score_fused_r2'}, {'score4_r2'},  {'score4f', 'score4b'});
	net.addLayer('score_pool3_r2', dagnn.Conv('size', [1 1 256 CLASSES], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool3_r2'}, {'score_pool3_r2'},  {'score_pool3f'  'score_pool3b'});
	net.addLayer('fusex_r2', dagnn.Sum(), {'score4_r2', 'score_pool3_r2'}, {'score_final_r2'}, {});
	net.addLayer('upsample_r2', dagnn.ConvTranspose('size', [8 8 CLASSES CLASSES], 'upsample', [8 8], 'hasBias', true, 'crop', [2 1 2 2]), {'score_final_r2'}, {'bigscore_r2'},  {'upsamplef', 'upsampleb'});
	
	net.addLayer('upsample_r2x', dagnn.ConvTranspose('size', [2 2 CLASSES CLASSES], 'upsample', [2 2], 'hasBias', true, 'crop', [0 0 0 0]), {'bigscore_r2'}, {'upsample_r2x'},  {'upsampler2x_f', 'upsampler2x_b'});
	net.addLayer('relu_upr2x', dagnn.ReLU(), {'upsample_r2x'}, {'relu_upr2x'}, {});
	net.addLayer('upsample_r2x2', dagnn.ConvTranspose('size', [2 2 CLASSES CLASSES], 'upsample', [2 2], 'hasBias', true, 'crop', [0 0 0 0]), {'relu_upr2x'}, {'upsample_r2x2'},  {'upsampler2x2_f', 'upsampler2x2_b'});
	net.addLayer('relu_upr2x2', dagnn.ReLU(), {'upsample_r2x2'}, {'relu_upr2x2'}, {});	

	net.addLayer('concat',  dagnn.Sum(), {'bigscore_r1', 'relu_upr2x2'}, {'concat'});
	net.addLayer('classifier', dagnn.Conv('size', [5 5 CLASSES CLASSES], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'concat'}, {'classifier'},  {'classf'  'classb'});

	net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
	net.addLayer('objective', dagnn.LossSemantic('weights', true), {'prob','label'}, 'objective');
	% -- end of the network

	% do the training!
	%initNet(net, [0.1, 1e-3*ones(1, 25)], 1e-3*ones(1, 25));
	initNet(net);
	net.conserveMemory = false;

	info = trainer(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train) ;
end


% function on charge of creating a batch of images + labels
function inputs = getBatch(opts, imdb, batch)
	images = imdb.images.data(:,:,:,batch) ;
	labels = imdb.images.labels(:, :, :, batch) ;
	if opts.useGpu > 0
  		images = gpuArray(images./127.0);
		labels = gpuArray(labels);
	end

	inputs = {'input', images, 'label', labels} ;
end

function initNet(net)
	net.initParams();
	%
	i = 1;
	for l=1:length(net.layers)
		% is a convolution layer?
		if(isa(net.layers(l).block, 'dagnn.Conv') || isa(net.layers(l).block, 'dagnn.ConvTranspose'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			%net.params(f_ind).value = F(i)*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;
            net.params(f_ind).trainMethod = 'adam';
            net.params(f_ind).m = zeros(size(net.params(f_ind).value), 'single');
            net.params(f_ind).v = zeros(size(net.params(f_ind).value), 'single');
            net.params(f_ind).t = 1;


			%net.params(b_ind).value = B(i)*randn(size(net.params(b_ind).value), 'single');
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

function W = orthoIn(s_)
	a = randn(s_(1)*s_(2)*s_(3), s_(4), 'single');
	[u,d,v] = svd(a, 'econ');
    if(size(a,1) < size(a, 2))
        u = v';
    end
	W = sqrt(2).*reshape(u, s_);
end

function initNetOrthogonal(net)
	net.initParams();
	%
	i = 1;
	for l=1:length(net.layers)
		% is a convolution layer?
		if(isa(net.layers(l).block, 'dagnn.Conv') || isa(net.layers(l).block, 'dagnn.ConvTranspose'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

            l
            size(net.params(f_ind).value)

			net.params(f_ind).value = orthoIn(size(net.params(f_ind).value));
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;
            net.params(f_ind).trainMethod = 'adam';
            net.params(f_ind).m = zeros(size(net.params(f_ind).value), 'single');
            net.params(f_ind).v = zeros(size(net.params(f_ind).value), 'single');
            net.params(f_ind).t = 1;


			net.params(b_ind).value = 1e-3*randn(size(net.params(b_ind).value), 'single');
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

function initNet2(net, netF)
	net.initParams();

	v = [1,3,6,8,11,13,15,18,20,22,25,27,29,32,34];
	%
	i = 1;
	for l=1:length(net.layers)
		% is a convolution layer?
		if(isa(net.layers(l).block, 'dagnn.Conv') || isa(net.layers(l).block, 'dagnn.ConvTranspose'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);
            
            if(i <= numel(v))
                net.params(f_ind).value = netF.layers{v(i)}.filters;
                net.params(f_ind).learningRate = 1e-6;
                net.params(f_ind).weightDecay = 1;
                
                 net.params(b_ind).value =  netF.layers{v(i)}.biases;
                 net.params(b_ind).learningRate = 1e-6;
                 net.params(b_ind).weightDecay = 1;
            else
                net.params(f_ind).trainMethod = 'adam';
			    net.params(f_ind).m = zeros(size(net.params(f_ind).value), 'single');
			    net.params(f_ind).v = zeros(size(net.params(f_ind).value), 'single');
			    net.params(f_ind).t = 1;
                
                net.params(b_ind).trainMethod = 'adam';
                net.params(b_ind).m = zeros(size(net.params(b_ind).value), 'single');
                net.params(b_ind).v = zeros(size(net.params(b_ind).value), 'single');
                net.params(b_ind).t = 1;
            end
	
			i = i + 1;
           
		end
	end
end
