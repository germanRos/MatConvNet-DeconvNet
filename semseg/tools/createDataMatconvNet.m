function [imdb] = createDataMatconvNet_Multi13()
	run(fullfile(fileparts(mfilename('fullpath')),'../../matlab/', 'vl_setupnn.m')) ;
	
 	%SIZE = [360, 480];
	SIZE = [180, 240];

	% CamVid Train
%	[imdb] = loadData('/home/german/Data/DL/CamVid', '/home/german/Data/DL/CamVid/train.txt', SIZE, 1);
%	% CamVid Test
%	[imdb2] = loadData('/home/german/Data/DL/CamVid', '/home/german/Data/DL/CamVid/test.txt', SIZE, 2);
%	imdb.images.data = cat(4, imdb.images.data, imdb2.images.data);
%	imdb.images.labels = cat(4, imdb.images.labels, imdb2.images.labels);
%	imdb.images.set = cat(2, imdb.images.set, imdb2.images.set);
%	clear imdb2;

%	
%	% KITTI Train
%	[imdb2] = loadData('/home/german/Data/DL/kitti_combined', '/home/german/Data/DL/kitti_combined/train.txt', SIZE, 3);
%	imdb.images.data = cat(4, imdb.images.data, imdb2.images.data);
%	imdb.images.labels = cat(4, imdb.images.labels, imdb2.images.labels);
%	imdb.images.set = cat(2, imdb.images.set, imdb2.images.set);
%	clear imdb2;
%	% KITTI Test
%	[imdb2] = loadData('/home/german/Data/DL/kitti_combined', '//home/german/Data/DL/kitti_combined/test.txt', SIZE, 4);
%	imdb.images.data = cat(4, imdb.images.data, imdb2.images.data);
%	imdb.images.labels = cat(4, imdb.images.labels, imdb2.images.labels);
%	imdb.images.set = cat(2, imdb.images.set, imdb2.images.set);
%	clear imdb2;

%	% LabelMe Train
%	[imdb2] = loadData_LABELME('/home/german/Data/DL/LabelMeDatasetToshiba', '/home/german/Data/DL/LabelMeDatasetToshiba/train.txt', SIZE, 5);
%	imdb.images.data = cat(4, imdb.images.data, imdb2.images.data);
%	imdb.images.labels = cat(4, imdb.images.labels, imdb2.images.labels);
%	imdb.images.set = cat(2, imdb.images.set, imdb2.images.set);
%	clear imdb2;
%	% LabelMe Test
%	[imdb2] = loadData_LABELME('/home/german/Data/DL/LabelMeDatasetToshiba', '/home/german/Data/DL/LabelMeDatasetToshiba/test.txt', SIZE, 6);
%	imdb.images.data = cat(4, imdb.images.data, imdb2.images.data);
%	imdb.images.labels = cat(4, imdb.images.labels, imdb2.images.labels);
%	imdb.images.set = cat(2, imdb.images.set, imdb2.images.set);
%	clear imdb2;

%	% CBCL Train
%	[imdb2] = loadData_CBCL('/home/german/Data/DL/cbcl_clean/semantic', '/home/german/Data/DL/cbcl_clean/semantic/train.txt', SIZE, 7);
%	imdb.images.data = cat(4, imdb.images.data, imdb2.images.data);
%	imdb.images.labels = cat(4, imdb.images.labels, imdb2.images.labels);
%	imdb.images.set = cat(2, imdb.images.set, imdb2.images.set);
%	clear imdb2;
%	% CBCL Test
%	[imdb2] = loadData_CBCL('/home/german/Data/DL/cbcl_clean/semantic', '/home/german/Data/DL/cbcl_clean/semantic/test.txt', SIZE, 8);
%	imdb.images.data = cat(4, imdb.images.data, imdb2.images.data);
%	imdb.images.labels = cat(4, imdb.images.labels, imdb2.images.labels);
%	imdb.images.set = cat(2, imdb.images.set, imdb2.images.set);
%	clear imdb2;

%	[imdb2] = loadData('/home/german/Data/DL/SynthCity', '/home/german/Data/DL/SynthCity/ALL.txt', SIZE, 9);
%	imdb.images.data = cat(4, imdb.images.data, imdb2.images.data);
%	imdb.images.labels = cat(4, imdb.images.labels, imdb2.images.labels);
%	imdb.images.set = cat(2, imdb.images.set, imdb2.images.set);
%	clear imdb2;

	[imdb] = loadData_CITYSCAPES('/media/IMLmatuser/405b226a-def9-479a-be6b-75c450d61555/gros/Data/cityscapes', '/media/IMLmatuser/405b226a-def9-479a-be6b-75c450d61555/gros/Data/cityscapes/all.txt', SIZE, 12);

	    imdb.images.data = vl_nnspnorm(imdb.images.data, [7,7, 1, 0.4]);
    
    % mean subtraction per channel per dataset
    for dset = 12
        ind = imdb.images.set==dset;
        N = size(imdb.images.data(:,:,:,ind),4);
        m = mean(reshape(permute(imdb.images.data(:,:,:,ind),[1 2 4 3]),[SIZE(2)*SIZE(1)*N],3));
        imdb.images.data(:,:,:,ind) = imdb.images.data(:,:,:,ind) - repmat(permute(m,[1 3 2]),[SIZE(1) SIZE(2) 1 sum(ind)]);
    end

    % observe stats
    % hist(imdb.images.data(:),100)

    % apply ceiling to the upper outliers in a hacky way specific to this dataset
    imdb.images.data = min(imdb.images.data,3);
    imdb.images.data = max(imdb.images.data,-3);
    
    % merge datasets 4 and 5 into 4 (both validation)
       
    
    % expand to fill the range
    range_min = min(imdb.images.data(:)); assert(range_min<0);
    range_max = max(imdb.images.data(:)); assert(range_max>0);
    range_multiplier = 127./max(abs(range_min),range_max);
    imdb.images.data = imdb.images.data.*range_multiplier;

end

function [imdbO] = shiftRange(imdb, s)
	i = imdb.images.data(:);
	minx = min(i);
	maxx = max(i);

	imdbO = imdb;
	imdbO.images.data = imdbO.images.data * s / (maxx - minx);
end

function [imdb]  = zeroMean(imdb)
    ids = unique(imdb.images.set);
    for i=ids
       indices = find(imdb.images.set == i); 
       a = imdb.images.data(:,:,:, indices);
       mm = mean(mean(mean(a, 4),1),2);
       imdb.images.data(:,:,1, indices) = imdb.images.data(:,:,1, indices) -mm(1);
       imdb.images.data(:,:,2, indices) = imdb.images.data(:,:,2, indices) -mm(2);
       imdb.images.data(:,:,3, indices) = imdb.images.data(:,:,3, indices) -mm(3);

    end
end

function [imdb] = loadData_CITYSCAPES(pathIn, file, newsize, id)
	fid = fopen(file);
	CFiles = textscan(fid, '%s');
	fclose(fid);

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

	% sizes
	RGBs = dir([pathIn '/RGB/*.png']);
	N = numel(CFiles{1});
	NL = 12;
	L = cell(1, NL);
	for i=1:NL
		L{1, i} = i;
	end

	stats = zeros(NL, 1);
	meta.sets = {'train', 'val', 'test'};
	meta.classes = L;

	% getting WxH
	im = imread([pathIn '/RGB/' RGBs(1).name]);
	if(~isempty(newsize))
		H = newsize(1);
		W = newsize(2);
		C = 3;	
	else
		[H, W, C] = size(im);
	end


	% zero initialization
	images.data = single(zeros(H, W, 3, N));
	images.data_mean = single(zeros(H, W, 3));
	images.labels = zeros(H, W, 2, N);
	images.set = zeros(1, N);

	for i=1:N
		IM = single(imread([pathIn, '/RGB/' CFiles{1}{i}]));
		strbase = strsplit(CFiles{1}{i}, '.');

		if(~isempty(newsize))
			images.data(:, :, :, i) = imresize(IM, newsize);
		else
			images.data(:, :, :, i) = IM;
		end

		labs = imread([pathIn, '/GT_CHAINER/' strbase{1} '.png']);
		if(~isempty(newsize))
			labs = imresize(labs, newsize, 'nearest');
		end

		indices0 = (labs(:) <= 0);
		labs(indices0) = 0;
		
		images.labels(:, :, 1, i) = labs;
		images.set(1, i) = id;

		for j=0:11
			ll = (labs == j);
			stats(j+1) = stats(j+1) + sum(ll(:));
		end


		str = sprintf('[%d / %d]', i, N);
		disp(str);
	end

	stats = stats ./ sum(stats);
	stats = median(stats) ./ stats;

	% time for adding extra channels
	aux = zeros(H, W);
	for i=1:size(images.labels, 4)
		for j=0:11
			ll = (images.labels(:, :, 1, i) == j);
			aux(ll) = stats(j+1);
		end

		images.labels(:, :, 2, i) = aux;
	end



	images.data_mean =  single(mean(images.data, 4));
	%images.data = bsxfun(@minus, images.data, images.data_mean) ;
	imdb.images = images;
	imdb.meta = meta;
	imdb.stats = stats;
end

  
function [imdb] = loadData_LABELME(pathIn, file, newsize, id)
	fid = fopen(file);
	CFiles = textscan(fid, '%s');
	fclose(fid);

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

	% sizes
	RGBs = dir([pathIn '/RGB/*.jpg']);
	N = numel(CFiles{1});
	NL = 12;
	L = cell(1, NL);
	for i=1:NL
		L{1, i} = i;
	end

	stats = zeros(NL, 1);
	meta.sets = {'train', 'val', 'test'};
	meta.classes = L;

	% getting WxH
	im = imread([pathIn '/RGB/' RGBs(1).name]);
	if(~isempty(newsize))
		H = newsize(1);
		W = newsize(2);
		C = 3;	
	else
		[H, W, C] = size(im);
	end


	% zero initialization
	images.data = single(zeros(H, W, 3, N));
	images.data_mean = single(zeros(H, W, 3));
	images.labels = zeros(H, W, 2, N);
	images.set = zeros(1, N);

	for i=1:N
		IM = single(imread([pathIn, '/RGB/' CFiles{1}{i}]));
		strbase = strsplit(CFiles{1}{i}, '.');

		if(~isempty(newsize))
			images.data(:, :, :, i) = imresize(IM, newsize);
		else
			images.data(:, :, :, i) = IM;
		end

		labs = imread([pathIn, '/GT/' strbase{1} '.png']);
		if(~isempty(newsize))
			labs = imresize(labs, newsize, 'nearest');
		end

		indices0 = (labs(:) <= 0);
		labs(indices0) = 0;
		
		images.labels(:, :, 1, i) = labs;
		images.set(1, i) = id;

		for j=0:11
			ll = (labs == j);
			stats(j+1) = stats(j+1) + sum(ll(:));
		end


		str = sprintf('[%d / %d]', i, N);
		disp(str);
	end

	stats = stats ./ sum(stats);
	stats = median(stats) ./ stats;

	% time for adding extra channels
	aux = zeros(H, W);
	for i=1:size(images.labels, 4)
		for j=0:11
			ll = (images.labels(:, :, 1, i) == j);
			aux(ll) = stats(j+1);
		end

		images.labels(:, :, 2, i) = aux;
	end



	images.data_mean =  single(mean(images.data, 4));
	%images.data = bsxfun(@minus, images.data, images.data_mean) ;
	imdb.images = images;
	imdb.meta = meta;
	imdb.stats = stats;
end

function [imdb] = loadData_CBCL(pathIn, file, newsize, id)
	fid = fopen(file);
	CFiles = textscan(fid, '%s');
	fclose(fid);

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

	% sizes
	RGBs = dir([pathIn '/RGB/*.jpg']);
	N = numel(CFiles{1});
	NL = 12;
	L = cell(1, NL);
	for i=1:NL
		L{1, i} = i;
	end

	stats = zeros(NL, 1);
	meta.sets = {'train', 'val', 'test'};
	meta.classes = L;

	% getting WxH
	im = imread([pathIn '/RGB/' RGBs(1).name]);

	if(~isempty(newsize))
		H = newsize(1);
		W = newsize(2);
		C = 3;	
	else
		[H, W, C] = size(im);
	end
	
	

	% zero initialization
	images.data = single(zeros(H, W, 3, N));
	images.data_mean = single(zeros(H, W, 3));
	images.labels = zeros(H, W, 2, N);
	images.set = zeros(1, N);

	for i=1:N
		IM = single(imread([pathIn, '/RGB/' CFiles{1}{i}]));
		strbase = strsplit(CFiles{1}{i}, '.');

		if(~isempty(newsize))
			images.data(:, :, :, i) = imresize(IM, newsize);
		else
			images.data(:, :, :, i) = IM;
		end

		labs = load([pathIn, '/GTTXT/' strbase{1} '.txt']);
		if(~isempty(newsize))
			labs = imresize(labs, newsize, 'nearest');
		end

		indices0 = (labs(:) <= 0);
		labs(indices0) = 0;
		
		images.labels(:, :, 1, i) = labs;
		images.set(1, i) = id;

		for j=0:11
			ll = (labs == j);
			stats(j+1) = stats(j+1) + sum(ll(:));
		end


		str = sprintf('[%d / %d]', i, N);
		disp(str);
	end

	stats = stats ./ sum(stats);
	stats = median(stats) ./ stats;

	% time for adding extra channels
	aux = zeros(H, W);
	for i=1:size(images.labels, 4)
		for j=0:11
			ll = (images.labels(:, :, 1, i) == j);
			aux(ll) = stats(j+1);
		end

		images.labels(:, :, 2, i) = aux;
	end

	%images.data = vl_nnspnorm(images.data, [7,7, 1, 0.5]); %images.data = vl_nnnormalize(images.data, [5 2 1.0000e-04 0.7500]);


	images.data_mean =  single(mean(images.data, 4));
	%images.data = bsxfun(@minus, images.data, images.data_mean) ;
	imdb.images = images;
	imdb.meta = meta;
	imdb.stats = stats;
end

function [imdb] = loadData(pathIn, file, newsize, id)
	fid = fopen(file);
	CFiles = textscan(fid, '%s');
	fclose(fid);

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

	% sizes
	RGBs = dir([pathIn '/RGB/*.png']);
	N = numel(CFiles{1});
	NL = 12;
	L = cell(1, NL);
	for i=1:NL
		L{1, i} = i;
	end

	stats = zeros(NL, 1);
	meta.sets = {'train', 'val', 'test'};
	meta.classes = L;

	% getting WxH
	im = imread([pathIn '/RGB/' RGBs(1).name]);

	if(~isempty(newsize))
		H = newsize(1);
		W = newsize(2);
		C = 3;	
	else
		[H, W, C] = size(im);
	end
	
	

	% zero initialization
	images.data = single(zeros(H, W, 3, N));
	images.data_mean = single(zeros(H, W, 3));
	images.labels = zeros(H, W, 2, N);
	images.set = zeros(1, N);

	for i=1:N
		IM = single(imread([pathIn, '/RGB/' CFiles{1}{i}]));
		strbase = strsplit(CFiles{1}{i}, '.');

		if(~isempty(newsize))
			images.data(:, :, :, i) = imresize(IM, newsize);
		else
			images.data(:, :, :, i) = IM;
		end

		labs = load([pathIn, '/GTTXT/' strbase{1} '.txt']);
		if(~isempty(newsize))
			labs = imresize(labs, newsize, 'nearest');
		end

		indices0 = (labs(:) <= 0);
		labs(indices0) = 0;
		
		images.labels(:, :, 1, i) = labs;
		images.set(1, i) = id;

		for j=0:11
			ll = (labs == j);
			stats(j+1) = stats(j+1) + sum(ll(:));
		end


		str = sprintf('[%d / %d]', i, N);
		disp(str);
	end

	stats = stats ./ sum(stats);
	stats = median(stats) ./ stats;

	% time for adding extra channels
	aux = zeros(H, W);
	for i=1:size(images.labels, 4)
		for j=0:11
			ll = (images.labels(:, :, 1, i) == j);
			aux(ll) = stats(j+1);
		end

		images.labels(:, :, 2, i) = aux;
	end

	%images.data = vl_nnspnorm(images.data, [7,7, 1, 0.5]); %images.data = vl_nnnormalize(images.data, [5 2 1.0000e-04 0.7500]);



	images.data_mean =  single(mean(images.data, 4));
	%images.data = bsxfun(@minus, images.data, images.data_mean) ;
	imdb.images = images;
	imdb.meta = meta;
	imdb.stats = stats;
end
