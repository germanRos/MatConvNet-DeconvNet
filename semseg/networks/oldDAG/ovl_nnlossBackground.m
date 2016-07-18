function Y = vl_nnlossBackground(X,c,dzdy)

	X = X + 1e-4 ;
	sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;
    c(:,:,2,:) = [] ;
	% indices of the elements that are 0
	c_ = find(c(:) == 0);
	[ir, ic, in] = ind2sub([sz(1), sz(2), sz(4)], c_);

	% now we get the indices in X for the given foreground classes
	% i.e., 7 8 9 10 11
	idxC7 = sub2ind(sz, ir, ic, 7*ones(numel(ir), 1), in);
	idxC8 = sub2ind(sz, ir, ic, 8*ones(numel(ir), 1), in);
	idxC9 = sub2ind(sz, ir, ic, 9*ones(numel(ir), 1), in);
	idxC10 = sub2ind(sz, ir, ic, 10*ones(numel(ir), 1), in);
	idxC11 = sub2ind(sz, ir, ic, 11*ones(numel(ir), 1), in);
	idxs = [idxC7; idxC8; idxC9; idxC10; idxC11];

	n = numel(idxs);
	if nargin <= 2
		Y = sum(X(idxs))./n;
    else
        Y = zeros(size(X));
        Y(idxs) = 1;
 		Y = Y * dzdy ./ n;
	end
end
