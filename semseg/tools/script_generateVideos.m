% base output folder
baseoutput = '/media/IMLmatuser/DATA/Results/YouTube';

% cell array of models
cmodels = {};cmodels{end+1} = '/home/IMLmatuser/MatConvNet-DeconvNet/semseg/tools/tnet_tksmpwce_bnormFULLT.mat';


% pre-loading all the nets
cnets = {};
for i=1:numel(cmodels)
	netx = load(cmodels{i});
	cnets{end+1} = netx.net;
end

% cell array with the video folders
cfolders = {};

cfolders{end+1} = '/media/IMLmatuser/DATA/YouTube/Compilation';

opts.H = 180; opts.W = 240;
for i=1:numel(cfolders)
	% process the dataset with all the nets
	for j=1:numel(cnets)
		[~, bn_, ~] = fileparts(cmodels{j});
        	[~, bn_data, ~] = fileparts(cfolders{i});
		outfolder = sprintf('%s/%s/%s', baseoutput, bn_data, bn_);
		
		
		if( (j==1) || (j==4) )
			opts.BSIZE = 15; % for tnets
		else
			opts.BSIZE = 10; % for FCNs
		end
		
		processVideoSeq(cfolders{i}, outfolder, cnets{j}, opts);

		str0 = sprintf('---- Finished net [%d / %d] of the dataset [%d / %d] ----', j, numel(cnets), i, numel(cfolders));
		disp(str0);
	end
end
