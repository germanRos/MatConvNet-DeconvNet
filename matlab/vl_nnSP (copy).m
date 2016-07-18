function [Y] = vl_nnSP(inputs, dzdy)
	BLOBS = inputs{1};
	SEGS  = inputs{2};
    M = max(SEGS(:));
	size_ = size(BLOBS);
    
    BLOBSP = permute(BLOBS, [1,2,4,3]);
    B2 = reshape(BLOBSP, size_(1)*size_(2), size_(4),11);
    SP = permute(SEGS, [1,2,4,3]);
    
    idx = (SP == 1);
    a = B2(idx(:),:);

	if nargin < 2
        Y = BLOBS*0;

        for i=1:size(SEGS, 4)
			blob = squeeze(BLOBS(:,:,:,i));
			seg = squeeze(SEGS(:,:,i));
			y = squeeze(Y(:,:,:,i));

			for s=0:max(seg(:))
				ind = find(seg == s);
				
				y(ind + 0*size_(1)*size_(2)) = mean(blob(ind + 0*size_(1)*size_(2)));
				y(ind + 1*size_(1)*size_(2)) = mean(blob(ind + 1*size_(1)*size_(2)));
				y(ind + 2*size_(1)*size_(2)) = mean(blob(ind + 2*size_(1)*size_(2)));
				y(ind + 3*size_(1)*size_(2)) = mean(blob(ind + 3*size_(1)*size_(2)));
				y(ind + 4*size_(1)*size_(2)) = mean(blob(ind + 4*size_(1)*size_(2)));
				y(ind + 5*size_(1)*size_(2)) = mean(blob(ind + 5*size_(1)*size_(2)));
				y(ind + 6*size_(1)*size_(2)) = mean(blob(ind + 6*size_(1)*size_(2)));
				y(ind + 7*size_(1)*size_(2)) = mean(blob(ind + 7*size_(1)*size_(2)));
				y(ind + 8*size_(1)*size_(2)) = mean(blob(ind + 8*size_(1)*size_(2)));
				y(ind + 9*size_(1)*size_(2)) = mean(blob(ind + 9*size_(1)*size_(2)));
				y(ind + 10*size_(1)*size_(2)) = mean(blob(ind + 10*size_(1)*size_(2)));

			end

			Y(:,:,:,i) = y;
		end
	else 
		Y = cell(1, 2);
        G = 1 + BLOBS*0;
        
         for i=1:size(SEGS, 4)
			seg = squeeze(SEGS(:,:,i));
			g = squeeze(G(:,:,:,i));

			for s=0:max(seg(:))
				ind = find(seg == s);
				N = 1/numel(ind);
                
				g(ind + 0*size_(1)*size_(2)) = N;
                g(ind + 1*size_(1)*size_(2)) = N;
                g(ind + 2*size_(1)*size_(2)) = N;
                g(ind + 3*size_(1)*size_(2)) = N;
                g(ind + 4*size_(1)*size_(2)) = N;
                g(ind + 5*size_(1)*size_(2)) = N;
                g(ind + 6*size_(1)*size_(2)) = N;
                g(ind + 7*size_(1)*size_(2)) = N;
                g(ind + 8*size_(1)*size_(2)) = N;
                g(ind + 9*size_(1)*size_(2)) = N;
                g(ind + 10*size_(1)*size_(2)) = N;

			end

			G(:,:,:,i) = g;
         end
        
        Y{1} = G.*dzdy;
	end
end

