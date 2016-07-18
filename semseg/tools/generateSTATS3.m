function [imdb, stats2] = generateSTATS3(imdb, id)
    indices = [];
	for i=1:numel(id)
		indx = find(imdb.images.set == id(i));
		indices = cat(2, indices, indx);
   end
   	
	labs = imdb.images.labels(:,:,1,:);
	uniqueLabs = unique(labs(:));
	minL = min(uniqueLabs);
	maxL = max(uniqueLabs);
	L = maxL - minL + 1;
	stats = zeros(L, 1);
	for j=minL:maxL
		ll = (imdb.images.labels(:,:,1,indices) == j);
		stats(j+1) = stats(j+1) + sum(ll(:));
    end
    
    st = 1./stats;
    
    new_min = 1;
    new_max = 50;

    p_range = max(st) - min(st);
    st = (st - min(st)) / p_range;
    n_range = new_max - new_min;
    stats2 = (st * n_range)  + new_min;
    
	W = stats2(labs+1);
	imdb.images.labels = cat(3, labs, W);
end
