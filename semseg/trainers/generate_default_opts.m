function opts = generate_default_opts()
	opts.expDir = '';
	opts.continue = true ;
	opts.batchSize = 256 ;
	opts.numSubBatches = 1 ;
	opts.train = [] ;
	opts.val = [] ;
	opts.gpus = [] ;
	opts.prefetch = false ;
	opts.numEpochs = 300 ;
	opts.learningRate = 0.001 ;
	opts.weightDecay = 0.0005 ;
	opts.momentum = 0.9 ;
	opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
	opts.profile = false ;
	opts.savePlots = true;

	% batch policies
	opts.bmanagerTrain = BatchManagerBase();
	opts.bmanagerVal = BatchManagerBase();

	% ADAM optimizer
	opts.alpha = 0.001;
    	opts.beta1 = 0.9;
    	opts.beta2 = 0.999;

	% stats and visualization
	opts.classesNames = {};
	opts.colorMapGT = [];
	opts.colorMapEst = [];
	opts.exampleIndices = [];
	opts.derOutputs = {'objective', 1} ;
	opts.extractStatsFn = @extract_stats_segmentation;
	opts.fcn_visualize = @visualize_segmentation;
    	opts.vis.hnd_loss = figure()
	opts.vis.hand_conf = figure()
	opts.vis.hand_examples = figure()
	% ----
end
