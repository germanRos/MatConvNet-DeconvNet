function [net, info] = FCN_DROP(imdb, netF, inpt, varargin)

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
     bopts_train.balanceStruct  = [1, 4; 2, 6; 12, 4];
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

	CLASSES = numel(opts.train.classesNames);

	% network definition
	net = dagnn.DagNN() ;
	
	% BLOCK 1 [CONV | RELU | CONV | RELU | M-POOL]
	net.addLayer('conv1_1', dagnn.Conv('size', [3 3 3 64], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'input'}, {'conv1_1'},  {'conv1_1f'  'conv1_1b'});
	net.addLayer('dropout1_1', dagnn.DropOut('rate', 0.1), {'conv1_1'}, {'conv1_1d'}, {});	
	net.addLayer('relu1_1', dagnn.ReLU(), {'conv1_1d'}, {'conv1_1x'}, {});
	net.addLayer('conv1_2', dagnn.Conv('size', [3 3 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv1_1x'}, {'conv1_2'},  {'conv1_2f'  'conv1_2b'});
	net.addLayer('dropout1_2', dagnn.DropOut('rate', 0.1), {'conv1_2'}, {'conv1_2d'}, {});		
	net.addLayer('relu1_2', dagnn.ReLU(), {'conv1_2d'}, {'conv1_2x'}, {});
	net.addLayer('pool1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv1_2x'}, {'pool1'}, {});

	% BLOCK 2 [CONV | RELU | CONV | RELU | M-POOL]
	net.addLayer('conv2_1', dagnn.Conv('size', [3 3 64 128], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool1'}, {'conv2_1'},  {'conv2_1f'  'conv2_1b'});
	net.addLayer('dropout2_1', dagnn.DropOut('rate', 0.1), {'conv2_1'}, {'conv2_1d'}, {});		
	net.addLayer('relu2_1', dagnn.ReLU(), {'conv2_1d'}, {'conv2_1x'}, {});
	net.addLayer('conv2_2', dagnn.Conv('size', [3 3 128 128], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv2_1x'}, {'conv2_2'},  {'conv2_2f'  'conv2_2b'});
	net.addLayer('dropout2_2', dagnn.DropOut('rate', 0.1), {'conv2_2'}, {'conv2_2d'}, {});		
	net.addLayer('relu2_2', dagnn.ReLU(), {'conv2_2d'}, {'conv2_2x'}, {});
	net.addLayer('pool2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv2_2x'}, {'pool2'}, {});
	
	% BLOCK 3 [CONV | RELU | CONV | RELU | CONV | RELU | M-POOL]
	net.addLayer('conv3_1', dagnn.Conv('size', [3 3 128 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool2'}, {'conv3_1'},  {'conv3_1f'  'conv3_1b'});
	net.addLayer('dropout3_1', dagnn.DropOut('rate', 0.1), {'conv3_1'}, {'conv3_1d'}, {});		
	net.addLayer('relu3_1', dagnn.ReLU(), {'conv3_1d'}, {'conv3_1x'}, {});
	net.addLayer('conv3_2', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv3_1x'}, {'conv3_2'},  {'conv3_2f'  'conv3_2b'});
	net.addLayer('dropout3_2', dagnn.DropOut('rate', 0.1), {'conv3_2'}, {'conv3_2d'}, {});		
	net.addLayer('relu3_2', dagnn.ReLU(), {'conv3_2d'}, {'conv3_2x'}, {});
	net.addLayer('conv3_3', dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv3_2x'}, {'conv3_3'},  {'conv3_3f'  'conv3_3b'});
	net.addLayer('dropout3_3', dagnn.DropOut('rate', 0.1), {'conv3_3'}, {'conv3_3d'}, {});		
	net.addLayer('relu3_3', dagnn.ReLU(), {'conv3_3d'}, {'conv3_3x'}, {});
	net.addLayer('pool3', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv3_3x'}, {'pool3'}, {});
	
	% BLOCK 4 [CONV | RELU | CONV | RELU | CONV | RELU | M-POOL]
	net.addLayer('conv4_1', dagnn.Conv('size', [3 3 256 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool3'}, {'conv4_1'},  {'conv4_1f'  'conv4_1b'});
	net.addLayer('dropout4_1', dagnn.DropOut('rate', 0.1), {'conv4_1'}, {'conv4_1d'}, {});		
	net.addLayer('relu4_1', dagnn.ReLU(), {'conv4_1d'}, {'conv4_1x'}, {});
	net.addLayer('conv4_2', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv4_1x'}, {'conv4_2'},  {'conv4_2f'  'conv4_2b'});
	net.addLayer('dropout4_2', dagnn.DropOut('rate', 0.1), {'conv4_2'}, {'conv4_2d'}, {});		
	net.addLayer('relu4_2', dagnn.ReLU(), {'conv4_2d'}, {'conv4_2x'}, {});
	net.addLayer('conv4_3', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv4_2x'}, {'conv4_3'},  {'conv4_3f'  'conv4_3b'});
	net.addLayer('dropout4_3', dagnn.DropOut('rate', 0.1), {'conv4_3'}, {'conv4_3d'}, {});	
	net.addLayer('relu4_3', dagnn.ReLU(), {'conv4_3d'}, {'conv4_3x'}, {});
	net.addLayer('pool4', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv4_3x'}, {'pool4'}, {});
	
	% BLOCK 5 [CONV | RELU | CONV | RELU | CONV | RELU | M-POOL]
	net.addLayer('conv5_1', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'pool4'}, {'conv5_1'},  {'conv5_1f'  'conv5_1b'});
	net.addLayer('dropout5_1', dagnn.DropOut('rate', 0.1), {'conv5_1'}, {'conv5_1d'}, {});		
	net.addLayer('relu5_1', dagnn.ReLU(), {'conv5_1d'}, {'conv5_1x'}, {});
	net.addLayer('conv5_2', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv5_1x'}, {'conv5_2'},  {'conv5_2f'  'conv5_2b'});
	net.addLayer('dropout5_2', dagnn.DropOut('rate', 0.1), {'conv5_2'}, {'conv5_2d'}, {});	
	net.addLayer('relu5_2', dagnn.ReLU(), {'conv5_2d'}, {'conv5_2x'}, {});
	net.addLayer('conv5_3', dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'stride', [1, 1], 'pad', [1, 1, 1, 1]), {'conv5_2x'}, {'conv5_3'},  {'conv5_3f'  'conv5_3b'});
	net.addLayer('dropout5_3', dagnn.DropOut('rate', 0.1), {'conv5_3'}, {'conv5_3d'}, {});	
	net.addLayer('relu5_3', dagnn.ReLU(), {'conv5_3d'}, {'conv5_3x'}, {});
	net.addLayer('pool5', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 1 0 1]), {'conv5_3x'}, {'pool5'}, {});
	
	% BLOCK 6 [CONV | RELU | DROPOUT ]
	net.addLayer('fc6', dagnn.Conv('size', [7 7 512 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool5'}, {'fc6'},  {'fc6f'  'fc6b'});
	net.addLayer('relu6', dagnn.ReLU(), {'fc6'}, {'fc6x'}, {});
	net.addLayer('dropout1', dagnn.DropOut('rate', 0.5), {'fc6x'}, {'fc6x2'}, {});

	
	% BLOCK 7 [CONV | RELU | DROPOUT | CONV]
	net.addLayer('fc7', dagnn.Conv('size', [1 1 4096 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'fc6x2'}, {'fc7'},  {'fc7f'  'fc7b'});
	net.addLayer('relu7', dagnn.ReLU(), {'fc7'}, {'fc7x'}, {});
	net.addLayer('dropout2', dagnn.DropOut('rate', 0.5), {'fc7x'}, {'fc7x2'}, {});
	net.addLayer('score_fr', dagnn.Conv('size', [1 1 4096 CLASSES], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'fc7x2'}, {'score'},  {'score_frf'  'score_frb'});
	
	% BLOCK 8 [ CONVT | CONV | CROP | SUM]
	net.addLayer('score2', dagnn.ConvTranspose('size', [4 4 CLASSES CLASSES], 'upsample', [2 2], 'hasBias', true, 'crop', [1 1 1 2], 'numGroups', 1), {'score'}, {'score2'},  {'score2f', 'score2b'});
	net.addLayer('score_pool4', dagnn.Conv('size', [1 1 512 CLASSES], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool4'}, {'score_pool4'},  {'score_pool4f'  'score_pool4b'});
	net.addLayer('fuse', dagnn.Sum(), {'score2', 'score_pool4'}, {'score_fused'}, {});

	% BLOCK 9 [ CONVT | CONV | CROP | SUM | CONVT | CROP]
	net.addLayer('score4', dagnn.ConvTranspose('size', [4 4 CLASSES CLASSES], 'upsample', [2 2], 'hasBias', true, 'crop', [2 1 1 1], 'numGroups', 1), {'score_fused'}, {'score4'},  {'score4f', 'score4b'});
	net.addLayer('score_pool3', dagnn.Conv('size', [1 1 256 CLASSES], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool3'}, {'score_pool3'},  {'score_pool3f'  'score_pool3b'});
	net.addLayer('fusex', dagnn.Sum(), {'score4', 'score_pool3'}, {'score_final'}, {});
	net.addLayer('upsample', dagnn.ConvTranspose('size', [8 8 CLASSES CLASSES], 'upsample', [8 8], 'hasBias', true, 'crop', [2 2 0 0]), {'score_final'}, {'bigscore'},  {'upsamplef', 'upsampleb'});

	net.addLayer('prob', dagnn.SoftMax(), {'bigscore'}, {'prob'}, {});
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

function initNet(net, F, B)
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
