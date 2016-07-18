function segments = visSP(im)
    im = im2single(im);
    
    segments = vl_slic(im, 5, 0.5, 'verbose') ;
    [sx,sy]=vl_grad(double(segments), 'type', 'forward') ;
    s = find(sx | sy) ;
    imp = im ;
    imp([s s+numel(im(:,:,1)) s+2*numel(im(:,:,1))]) = 0 ;
    figure; imagesc(imp);
end