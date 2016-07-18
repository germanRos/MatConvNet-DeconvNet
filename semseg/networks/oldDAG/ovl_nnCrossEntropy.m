function Y = vl_nnCrossEntropy(X,c,dzdy)
	% no division by zero
	X = X + 1e-4 ;

	if nargin <= 2
		d = -(c .* log(X));
		Y = sum(d(:));
	else
		Y = - (dzdy * c) ./ max(X, 1e-8) ;
	end	
end
