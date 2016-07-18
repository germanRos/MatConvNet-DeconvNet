function [Y, GT] = vl_nnbsplit(X, x_ids, trainDoms, GT, dzdy)
	cellx = arrayfun(@(a) x_ids == a, trainDoms, 'UniformOutput', false);
    	n = 1:size(X, 4);
	trainIdxs = n(any(cell2mat(cellx), 1));
	if nargin == 4
		if(numel(trainIdxs) ~= 0)
			Y = X(:,:,:, trainIdxs);
			GT = GT(:,:,:,trainIdxs);
        else
            Y = X;
            GT = GT;
		end 
	else
		Y = X*0;
		Y(:,:,:,trainIdxs) = dzdy;
        GT = GT;
	end
end
