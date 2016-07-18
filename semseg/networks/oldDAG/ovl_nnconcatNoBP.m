function [ Y ] = vl_nnconcatNoBP(X1, X2, dzdy)
	if(nargin == 2)
		Y = cat(3, X1, X2);
	else
		Y = cell(1, 2);
		Y{1} = gpuArray.zeros(size(X1), 'single');
		Y{2} = gpuArray.zeros(size(X2), 'single');
	end
end
