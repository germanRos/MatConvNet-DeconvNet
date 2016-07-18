function y = vl_nnLp(x, c, p, dzdy)
	d = bsxfun(@minus, x, c);
    
    S = 1./[ 1.0000
    1.0000
    2.0000
    3.5000
    7.5000
    2.0000
    5.5000
   10.0000
    7.0000
   10.0000
   14.0000
   14.0000];

for j=1:size(d, 3)
   d(:,:,j,:) = S(j) .* d(:,:,j,:); 
end

	% forward mode
	if isempty(dzdy)
        d = d(:);
        
		% L1
		if p == 1
			y = sum(abs(d));
		elseif p == 2
			y = sum(d.*d);
		else
			y = sum(abs(d).^p) ;
		end
	% backward mode
	else
		if p == 1
			y = bsxfun(@times, dzdy, sign(d)) ;
		elseif p == 2
			y = bsxfun(@times, 2 * dzdy, d) ;
		elseif p < 1
			y = bsxfun(@times, p * dzdy, max(abs(d), opts.epsilon).^(p-1) .* sign(d)) ;
		else
			y = bsxfun(@times, p * dzdy, abs(d).^(p-1) .* sign(d)) ;
		end
	end
end
