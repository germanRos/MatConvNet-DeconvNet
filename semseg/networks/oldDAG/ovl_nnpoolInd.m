function [Y, Ind, s] = vl_nnpoolInd(X, POOL, Ind, pad, varargin)
	% forward mode
	if(isempty(varargin))
		X_CPU = gather(X);

		% initial padding
		X1 = [zeros(pad(1), size(X_CPU, 2), size(X_CPU, 3), size(X_CPU, 4)); X_CPU; zeros(pad(2), size(X_CPU, 2), size(X_CPU, 3), size(X_CPU, 4))];
		XP = [zeros(size(X1, 1), pad(3), size(X1, 3), size(X1, 4)), X1, zeros(size(X1, 1), pad(4), size(X1, 3), size(X1, 4))];
	
		[mg, Ind] = MaxPooling(XP, single(POOL));
        Y = gpuArray(mg);	
        s = size(X);
	% backward mode
	else
		DZDX = varargin{1};

		Y = zeros(size(X));
		Y(Ind) = gather(DZDX);
		Y = gpuArray(Y((pad(1)+1):(end-pad(2)), (pad(3)+1):(end-pad(4)), :, :));
		Ind = [];
        s = [];
	end
end
