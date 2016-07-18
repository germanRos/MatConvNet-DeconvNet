function s=super_pixels(im)
     im = im - min(im(:));
    im = im ./ max(im(:));

    s = vl_slic(im, 5, 0.4) ;
end
