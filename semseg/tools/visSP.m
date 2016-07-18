function segments = visSP(im)
    im = im - min(im(:));
    im = im ./ max(im(:));
    im2 = single(vl_xyz2lab(vl_rgb2xyz(im)));
    segments = vl_slic(im2, 8, 2.5) ;
    [sx,sy]=vl_grad(double(segments), 'type', 'forward') ;
    s = find(sx | sy) ;
    imp = im ;
    imp([s s+numel(im(:,:,1)) s+2*numel(im(:,:,1))]) = 0.3 ;
    %figure; 
    imagesc(imp);
end