clc
clear all
close all

numDatasets = 5;
dataset = ["sixray", "gdxray", "compass", "opixray", "mvtec"];

for d = 1:numDatasets
    dataset(d)
    pn2 = join(['datasets\' dataset(d) '\input\']);
    pn2 = replace(pn2, ' ', '');

	pn = join(['datasets\' dataset(d) '\abnormal\']);
    pn = replace(pn, ' ', '');

    pn3 = join([pn2 '*.png']);
    pn3 = replace(pn3, ' ', '');
    ext_img = pn3;
    
    a = dir(ext_img);
    nfile = length(a);
    proposals = {};
    obj = [];

    for k=1:nfile
        fn = a(k).name; 
        
        pn4 = join([pn2 fn]);
        pn4 = replace(pn4, ' ', '');

        s = imread(pn4);
        
        pn5 = join([pn fn]);
        pn5 = replace(pn5, ' ', '');
        
        if(size(s,3) ~= 3)
            s = cat(3,s,s,s);
        end

        s = imresize(s,[2240 2240],'bilinear');

        t = imread('P00002.jpg'); % image used as a source for stylization

        if(size(t,3) ~= 3)
            t = cat(3,t,t,t);
        end

        t = imresize(t,[2240 2240],'bilinear');

        [r,c,ch] = size(t);

        s = imresize(s,[r,c],'bilinear');

        if ismatrix(s)
            s = cat(3,s,s,s);
        end

        ab = fft2(s);

        ab1 = fftshift(ab);
    %     ab1 = ab;
        abR = abs(ab1);
        abI = angle(ab1);

        ac = fft2(t);

        ac2 = fftshift(ac);
    %     ac2 = ac;
        acR = abs(ac2);
        acI = angle(ac2);

        [r,c,ch] = size(t);

        [r1,c1,ch1] = size(s);

        r = min(r,r1);
        c = min(c,c1);

        r = floor(r/2);
        c = floor(c/2);

        b = 1;

    %     abR(r-b:r+b,c-b:c+b,:) = acR(r-b:r+b,c-b:c+b,:);

        windowSize = [r1, c1];
        si = 1;

        G = fspecial('gaussian', windowSize, si);

    %     imshow(G,[])

        G = (G-min(min(G)))/(max(max(G))-min(min(G)));

        S = acR .* G;

        abR = abR + S;

        ab2 = abR .* exp(abI * 1i);
    %     ab2 = abR + abI * 1i;

        ab2 = ifftshift(ab2);

        ab2 = real(ifft2(ab2));

        ab2 = mat2gray(ab2);
        imshow(ab2);

        ab2 = imresize(ab2,[2240 2240],'bilinear');

        imwrite(ab2,pn5,'PNG');
    end
end