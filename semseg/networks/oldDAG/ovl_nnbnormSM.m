function [y,dzdg,dzdb] = vl_nnbnormSM(x,g,b, mode, mu_, sigma_, dzdy)
	epsilon = 1e-4 ;

	x_size = [size(x,1), size(x,2), size(x,3), size(x,4)];
	g_size = size(g) ;
	b_size = size(b) ;
	g = reshape(g, [1 x_size(3) 1]) ;
	b = reshape(b, [1 x_size(3) 1]) ;
	x = reshape(x, [x_size(1)*x_size(2) x_size(3) x_size(4)]) ;
	mass = prod(x_size([1 2 4])) ;

	% standard training/validation forward
	if(mode == 1)
        try
            mu = sum(sum(x,1),3) / mass ;
        catch ME
            mu = sum(sum(x,1),3) / mass ;
        end
		
		y = bsxfun(@minus, x, mu);
		sigma2 = sum(sum(y .* y,1),3) / mass + epsilon ;
		sigma = sqrt(sigma2) ;
		y = bsxfun(@plus, bsxfun(@times, g ./ sigma, y), b) ;
		y = reshape(y, x_size) ;
	% standard training/validation backward
	elseif(mode == 2)
		mu = sum(sum(x,1),3) / mass ;
		y = bsxfun(@minus, x, mu);
		sigma2 = sum(sum(y .* y,1),3) / mass + epsilon ;
		sigma = sqrt(sigma2) ;
		dzdy = reshape(dzdy, size(x)) ;
  		dzdg = sum(sum(dzdy .* y,1),3) ./ sigma ;
  		dzdb = sum(sum(dzdy,1),3) ;

  		muz = dzdb / mass;
		y = bsxfun(@times, g ./ sigma, bsxfun(@minus, dzdy, muz)) - bsxfun(@times, g .* dzdg ./ (sigma2 * mass), y) ;

  		dzdg = reshape(dzdg, g_size) ;
  		dzdb = reshape(dzdb, b_size) ;
		y = reshape(y, x_size) ;
	% fix training/validation forward
	elseif(mode == 3)
		mu = mu_;
		y = bsxfun(@minus, x, mu);
		sigma = sigma_;
		y = bsxfun(@plus, bsxfun(@times, g ./ sigma, y), b) ;
		y = reshape(y, x_size) ;
	% fix training/validation backward
	elseif(mode == 4)
		mu = mu_;
		sigma = sigma_;
		y = bsxfun(@minus, x, mu);
		sigma2 = sigma .* sigma;

		dzdy = reshape(dzdy, size(x)) ;
  		dzdg = sum(sum(dzdy .* y,1),3) ./ sigma ;
  		dzdb = sum(sum(dzdy,1),3) ;

  		muz = dzdb / mass;
		y = bsxfun(@times, g ./ sigma, bsxfun(@minus, dzdy, muz)) - bsxfun(@times, g .* dzdg ./ (sigma2 * mass), y) ;

  		dzdg = reshape(dzdg, g_size) ;
  		dzdb = reshape(dzdb, b_size) ;
		y = reshape(y, x_size) ;

	end


end
