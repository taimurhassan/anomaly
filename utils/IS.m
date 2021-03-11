clc
clear all
close all

load classifier2.mat

pn = 'C:\jaihc\datasets\gdxray\disc\heat\';
pn2 = 'datasets\gdxray\disc\real\';
pn3 = 'datasets\gdxray\disc\results\';

imagefiles = dir([pn '*.jpg']);

color = [194 187 194];
nfiles = length(imagefiles);    
k = 0;
for ii=1:1:nfiles
    fn = imagefiles(ii).name;
    img=imread([pn fn]);

    [r,c,ch] = size(img);
    mask = zeros(r,c);
    
    t = 20;
    for i = 224:224:c
        mask(:, i-t:i+t) = 255;
        mask = mask';
        mask(:, i-t:i+t) = 255;
    end
    
    mask = logical(mask(1:r,1:c));
    
    t = 50;
    for i = 1:r
        for j = 1:c
            s = abs(double(img(i,j,1)) - double(color(1)));
            s = s + abs(double(img(i,j,2)) - double(color(2)));
            s = s + abs(double(img(i,j,3)) - double(color(3)));
            if s > t
                img(i,j,1) = 0;
                img(i,j,2) = 0;
                img(i,j,3) = 0;
            end
        end
    end
    
    img = rgb2gray(img);
    
	img = regionfill(img,mask);
    
    i2 = imread([pn2 fn]);
    
    o = i2;
    
    numRegions = 2;
    [L,Centers] = imsegkmeans(img,numRegions);
    
    img = L;

    [r,c] = size(img);
	
    mask = logical(zeros(r,c));

    mask(img == 1) = 1;

    mask = bwareaopen(mask, 15000);
        
    mask = imfill(mask,'holes');
    
	s1 = sum(sum(mask==1));
            
    mask2 = logical(zeros(r,c));

    mask2(img == 2) = 1;

    mask2 = bwareaopen(mask2, 15000);
        
    mask2 = imfill(mask2,'holes');
    
	s2 = sum(sum(mask2==1));

    if s2 < s1
        mask = mask2;
    end
    
    cc = bwconncomp(mask); 
    bboxes = regionprops(cc,'BoundingBox');

    i3 = i2;

    for j = 1:length(bboxes)
        patch = imcrop(o, bboxes(j).BoundingBox); 
        patch = imresize(patch, [224 224],'bilinear');

        class = classify(classifier2, patch);
        
        if class == 'misc'
            continue;
        end
        
        i3 = insertObjectAnnotation(i3, 'rectangle', bboxes(j).BoundingBox, class,'LineWidth',30,'FontSize',72);
        
        patch2 = imcrop(mask, bboxes(j).BoundingBox);
        
        patch2 = bwareaopen(patch2,45500);
        
        patch2 = imfill(patch2,'holes');
        
        [row,col] =size(patch2);
        mask2 = zeros(size(mask));
        
        x1 = round(bboxes(j).BoundingBox(2));
        x2 = round(bboxes(j).BoundingBox(2)+bboxes(j).BoundingBox(4));
        
        if (x2-x1 ~= row)
            if x2-x1 < row
                x1 = x1 + 1;
            else
                x1 = x1 - 1;
            end
        end
        
        y1 = round(bboxes(j).BoundingBox(1));
        y2 = round(bboxes(j).BoundingBox(1)+bboxes(j).BoundingBox(3));
        
        if (y2-y1 ~= col)
            if y2-y1 < col
                y1 = y1 + 1;
            else
                y1 = y1 - 1;
            end
        end        

        mask2(x1:x2+(row-length(x1:x2)),y1:y2+(col-length(y1:y2)))=patch2;        
        
        a = i3(:,:,1);
        a(mask2 == 1) = 255;
        i3(:,:,1) = a;
        
        k = k + 1;
    end
    
    h = imshowpair(o, i3, 'montage');
        
    imwrite(h.CData,[pn3 fn],'JPEG');
 
end

function [Sxx, Sxy, Syy] = structureTensor(I,si,so)
I = double(I);
[m n] = size(I);
 
Sxx = NaN(m,n);
Sxy = NaN(m,n);
Syy = NaN(m,n);
 
x  = -2*si:2*si;
g  = exp(-0.5*(x/si).^2);
g  = g/sum(g);
gd = -x.*g/(si^2); 
 
Ix = conv2( conv2(I,gd,'same'),g','same' );
Iy = conv2( conv2(I,gd','same'),g,'same' );
 
Ixx = Ix.^2;
Ixy = Ix.*Iy;
Iyy = Iy.^2;
 
x  = -2*so:2*so;
g  = exp(-0.5*(x/so).^2);
Sxx = conv2( conv2(Ixx,g,'same'),g','same' ); 
Sxy = conv2( conv2(Ixy,g,'same'),g','same' );
Syy = conv2( conv2(Iyy,g,'same'),g','same' );

end

%% Hysteresis3D

function [tri,hys]=hysteresis3d(img,t1,t2,conn)
if nargin<3
    disp('function needs at least 3 inputs')
    return;
elseif nargin==3
    disp('inputs=3')
    if numel(size(img))==2;
        disp('img=2D')
        disp('conn set at 4 connectivies (number of neighbors)')
        conn=8;
    end
    if numel(size(img))==3; 
        disp('img=3D')
        disp('conn set at 6 connectivies (number of neighbors)')
        conn=6;
    end
end

if t1>t2    
	tmp=t1;
	t1=t2; 
	t2=tmp;
end
minv=min(img(:));               
maxv=max(img(:));                
t1v=t1*(maxv-minv)+minv;
t2v=t2*(maxv-minv)+minv;

tri=zeros(size(img));
tri(img>=t1v)=1;
tri(img>=t2v)=2;

abovet1=img>t1v;                                     
seed_indices=sub2ind(size(abovet1),find(img>t2v));   
hys=imfill(~abovet1,seed_indices,conn);              
hys=hys & abovet1;
end