function [net, info] = FCN_ENSEMBLE_FS(imdb, netF, inpt, varargin)
    
	% some common options
	trainer = @cnn_train_dag_seg_adv;

	opts.train.extractStatsFn = @extract_stats_segmentation;
	opts.train.batchSize = 6;
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
    	trainData2 = find(imdb.images.set == 2);   
    	trainData = cat(2, trainData1, trainData2);
    	valData = find(imdb.images.set == 3);

    valData = valData(1:100);
	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];

     % defining batch policies 
     bopts_train.mode           = 'balanced';
     bopts_train.imdb           = imdb;
     bopts_train.balanceStruct  =  [1, 3; 2, 7]; %[1, 6; 2, 2];
     bopts_train.maxNumBatches  = 250;
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

	% network definition
	net = dagnn.DagNN() ;
	net.addLayer('FCN_D', dagnn.FrozenNet('pathmodel', '/DATA/Results/NewNetworks/FCN_D.mat', 'output_point', 'upsample'), {'input'}, {'FCN_D'});
	net.addLayer('FCN_S', dagnn.FrozenNet('pathmodel', '/DATA/Results/NewNetworks/FCN_S.mat', 'output_point', 'upsample'), {'input'}, {'FCN_S'});
	net.addLayer('CONCAT', dagnn.ConcatMultiF('dim', 3, 'numFeats', [11, 6]), {'FCN_D', 'FCN_S'}, {'CONCAT'});

	net.addLayer('conv1', dagnn.Conv('size', [1, 1, 17, 128], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'CONCAT'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('relu1', dagnn.ReLU('leak', 0.001), {'conv1'}, {'relu1'}, {});
	
	net.addLayer('conv2_1', dagnn.Conv('size', [1, 1, 128, 128], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu1'}, {'conv2_1'},  {'conv2_1f'  'conv2_1b'});
	net.addLayer('relu2_1', dagnn.ReLU('leak', 0.001), {'conv2_1'}, {'relu2_1'}, {});
	net.addLayer('conv2_2', dagnn.Conv('size', [1, 1, 128, 128], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu2_1'}, {'conv2_2'},  {'conv2_2f'  'conv2_2b'});
	net.addLayer('sum2', dagnn.Sum(), {'relu1', 'conv2_2'}, {'sum2'}, {});
	net.addLayer('relu2_2', dagnn.ReLU('leak', 0.001), {'sum2'}, {'relu2_2'}, {});
	net.addLayer('drop2', dagnn.DropOut('rate', 0.5), {'relu2_2'}, {'drop2'});

	net.addLayer('conv3', dagnn.Conv('size', [3, 3, 128, 64], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'drop2'}, {'conv3'},  {'conv3f'  'conv3b'});
	net.addLayer('conv3_1', dagnn.Conv('size', [1, 1, 64, 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'conv3'}, {'conv3_1'},  {'conv3_1f'  'conv3_1b'});
	net.addLayer('relu3_1', dagnn.ReLU('leak', 0.001), {'conv3_1'}, {'relu3_1'}, {});
	net.addLayer('conv3_2', dagnn.Conv('size', [1, 1, 64, 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu3_1'}, {'conv3_2'},  {'conv3_2f'  'conv3_2b'});
	net.addLayer('sum3', dagnn.Sum(), {'conv3_2', 'conv3'}, {'sum3'}, {});
	net.addLayer('relu3_2', dagnn.ReLU('leak', 0.001), {'sum3'}, {'relu3_2'}, {});

	net.addLayer('conv4_1', dagnn.Conv('size', [1, 1, 64, 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu3_2'}, {'conv4_1'},  {'conv4_1f'  'conv4_1b'});
	net.addLayer('relu4_1', dagnn.ReLU('leak', 0.001), {'conv4_1'}, {'relu4_1'}, {});
	net.addLayer('conv4_2', dagnn.Conv('size', [1, 1, 64, 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu4_1'}, {'conv4_2'},  {'conv4_2f'  'conv4_2b'});
	net.addLayer('sum4', dagnn.Sum(), {'conv4_2', 'relu3_2'}, {'sum4'}, {});
	net.addLayer('relu4_2', dagnn.ReLU('leak', 0.001), {'sum4'}, {'relu4_2'}, {});

	net.addLayer('conv5_1', dagnn.Conv('size', [1, 1, 64, 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu4_2'}, {'conv5_1'},  {'conv5_1f'  'conv5_1b'});
	net.addLayer('relu5_1', dagnn.ReLU('leak', 0.001), {'conv5_1'}, {'relu5_1'}, {});
	net.addLayer('conv5_2', dagnn.Conv('size', [1, 1, 64, 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu5_1'}, {'conv5_2'},  {'conv5_2f'  'conv5_2b'});
	net.addLayer('sum5', dagnn.Sum(), {'conv5_2', 'relu4_2'}, {'sum5'}, {});
	net.addLayer('relu5_2', dagnn.ReLU('leak', 0.001), {'sum5'}, {'relu5_2'}, {});


	net.addLayer('classi', dagnn.Conv('size', [1, 1, 64, 11], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu5_2'}, {'classi'},  {'classi_1f'  'classi_1b'});
	net.addLayer('prob', dagnn.SoftMax(), {'classi'}, {'prob'}, {});
	net.addLayer('objective', dagnn.LossSemantic('weights', true), {'prob','label'}, 'objective');
	% -- end of the network

	% do the training!
    initNet(net);
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


function initNet(net)
	net.initParams();
	%
	for l=1:length(net.layers)
		% is a convolution layer?
		if(isa(net.layers(l).block, 'dagnn.Conv') || isa(net.layers(l).block, 'dagnn.ConvTranspose'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			net.params(f_ind).trainMethod = 'adam';
			net.params(f_ind).m = zeros(size(net.params(f_ind).value), 'single');
			net.params(f_ind).v = zeros(size(net.params(f_ind).value), 'single');
			net.params(f_ind).t = 1;


	
			net.params(b_ind).trainMethod = 'adam';
			net.params(b_ind).m = zeros(size(net.params(b_ind).value), 'single');
			net.params(b_ind).v = zeros(size(net.params(b_ind).value), 'single');
			net.params(b_ind).t = 1;
		end
	end
end

