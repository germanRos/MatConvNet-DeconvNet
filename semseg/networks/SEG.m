function [net, info] = SEG(imdb, netF, inpt, varargin)
    
	% some common options
	trainer = @cnn_train_dag_seg_adv;

	opts.train.extractStatsFn = @extract_stats_segmentation;
	opts.train.fcn_visualize = @visualize_segmentation_edges;
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
	trainData = find((imdb.images.set == 1) ); 
    	valData = find(imdb.images.set == 3);

    valData = valData(1:100);
	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];

     % defining batch policies 
     bopts_train.mode           = 'balanced';
     bopts_train.imdb           = imdb;
     bopts_train.balanceStruct  =  [1, 3]; %[1, 6; 2, 2];
     bopts_train.maxNumBatches  = 250;
     bmanagerTrain              = BatchManagerBalanced(bopts_train);
     
     bopts_val.mode             = 'seq';
     bopts_val.domains          = [3];
     bopts_val.bsize            = 3;
     bopts_val.imdb             = imdb;
     bmanagerVal                = BatchManagerSeq(bopts_val);
     
     opts.train.bmanagerTrain = bmanagerTrain;
     opts.train.bmanagerVal   = bmanagerVal;
   
	opts.train.classesNames = {'smooth', 'edge'};
	colorMap  = (1/255)*[		    
					    128 128 128
					    128 0 0
					    ];
	opts.train.colorMapGT = [0 0 0; colorMap];
	opts.train.colorMapEst = colorMap;

	% network definition
	net = dagnn.DagNN() ;
	net.addLayer('VGG16', dagnn.FrozenNet('pathmodel', '/home/IMLmatuser/MatConvNet-DeconvNet/semseg/smallNet.mat', 'output_point', 'x4'), {'input'}, {'VGG16'});
	net.addLayer('conv1', dagnn.Conv('size', [7, 7, 64, 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'VGG16'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('bn1', dagnn.BatchNorm('numChannels', 64), {'conv1'}, {'bn1'}, {'bn1f', 'bn1b', 'bn1m'});
	net.addLayer('relu1', dagnn.ReLU('leak', 0.001), {'bn1'}, {'relu1'}, {});
	net.addLayer('conv2', dagnn.Conv('size', [7, 7, 64, 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'relu1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('bn2', dagnn.BatchNorm('numChannels', 64), {'conv2'}, {'bn2'}, {'bn2f', 'bn2b', 'bn2m'});
	net.addLayer('relu2', dagnn.ReLU('leak', 0.001), {'bn2'}, {'relu2'}, {});

	net.addLayer('regressor', dagnn.Conv('size', [7, 7, 64, 2], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'relu2'}, {'regressor'},  {'regressor_1f'  'regressor_1b'});
	net.addLayer('prob', dagnn.SoftMax(), {'regressor'}, {'prob'}, {});
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
	labels = imdb.images.labels(:, :, :, batch);
	for l=1:size(labels, 4)
		labels(:,:,1,l) = (edge(squeeze(labels(:,:,1,l)),'sobel', 0.5))+1;
	end
	LL = labels(:,:,1,:);
	WW = labels(:,:,2,:);
	WW(:) = 1.0;
	WW(LL > 1) = 10.0;
	labels(:,:,2,:) = WW;
	

	if opts.useGpu > 0
  		images = gpuArray(images./127.0);
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

