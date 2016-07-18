function Y = vl_nnlossDA(X,c, sets, alpha, dzdy)
	idxSRC = (sets == 1);
	X_SRC = X(:,:,:,idxSRC);
	c_SRC = c(:,:,:,idxSRC);
	
	idxTRG = (sets == 3);
	X_TRG = X(:,:,:,idxTRG);
	c_TRG = c(:,:,:,idxTRG);

	% validation mode
	if( (sum(idxSRC)+sum(idxTRG)) == 0 )
		Y = 0;
	else
		cost_SRC = vl_nnloss(X_SRC, c_SRC);	
		cost_TRG = vl_nnloss(X_TRG, c_TRG);
	end

	if nargin <= 4		
		if( (sum(idxSRC)+sum(idxTRG)) ~= 0 )
			Y = 0.5*alpha * norm(cost_SRC - cost_TRG, 2)^2;
		end
	else
		Y_SRC_p = vl_nnloss(X_SRC, c_SRC, ones(1, 1));	
		Y_TRG_p = vl_nnloss(X_TRG, c_TRG, ones(1, 1));

		Y_SRC = gpuArray.zeros(size(X));
		Y_SRC(:,:,:, idxSRC) = Y_SRC_p;

		Y_TRG = gpuArray.zeros(size(X));
		Y_TRG(:,:,:, idxTRG) = Y_TRG_p;
		

		Y = alpha * (cost_SRC - cost_TRG) * (Y_SRC - Y_TRG) * dzdy;	
	end
end
