function Y = vl_nnCrossEntropyWeights(X,c,w,dzdy)
	% no division by zero
	X = X + 1e-4 ;

	[~, labs] = max(c, [], 3);
	W = w(labs);

 	%W = repmat(reshape(w,1,1, size(c, 3)), [size(c,1), size(c,2), 1,size(c,4)]);
	if nargin <= 3
		d = -(c .* log(X));
		dsum = sum(d, 3);
		dsum = W .* dsum;
		
		Y = sum(dsum(:));
	else
		Y = - (dzdy .* bsxfun(@times, W, c)) ./ max(X, 1e-8) ;
	end	
end
