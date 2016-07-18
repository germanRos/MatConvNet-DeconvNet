function [Y, Ind, s] = vl_nnpoolInd3(X, POOL, Ind, pad, Kernel, varargin)
	% forward mode
	if(isempty(varargin))
		% initial padding
		X1 = [gpuArray.zeros(pad(1), size(X, 2), size(X, 3), size(X, 4), 'single'); X; gpuArray.zeros(pad(2), size(X, 2), size(X, 3), size(X, 4), 'single')];
		XP = [gpuArray.zeros(size(X1, 1), pad(3), size(X1, 3), size(X1, 4), 'single'), X1, gpuArray.zeros(size(X1, 1), pad(4), size(X1, 3), size(X1, 4), 'single')];
		Y = gpuArray.zeros(ceil(size(XP, 1)/POOL), ceil(size(XP, 2)/POOL), size(XP, 3), size(XP, 4), 'single');
		Ind = gpuArray.zeros(numel(Y), 1, 'int32');

		[Y, Ind] = feval(Kernel, Y, Ind, XP, int32(size(XP, 1)), int32(size(XP, 2)), int32(size(XP, 3)), int32(size(XP, 4)), POOL);
        	s = size(X);
	% backward mode
	else
		DZDX = varargin{1};

		Y = gpuArray.zeros(size(X), 'single');
		Y(Ind+1) = DZDX;
		Y = (Y((pad(1)+1):(end-pad(2)), (pad(3)+1):(end-pad(4)), :, :));
		Ind = [];
        	s = [];
	end
end
