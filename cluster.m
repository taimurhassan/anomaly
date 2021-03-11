clc
clear all
close all

dataset = ["sixray", "gdxray", "compass", "opixray"];
numClusters = [4, 3, 3, 4];

color{1} = [214 255 230]; 
color{2} = [255 221 255]; 
color{3} = [0 255 0]; 
color{4} = [0 255 0]; 

for d = 1:length(dataset)
    dataset(d)
    pn = join(['datasets\' dataset(d) '\results\fake\']);
    pn = replace(pn, ' ', '');
    pn2 = join(['datasets\' dataset(d) '\results\real\']);
    pn2 = replace(pn2, ' ', '');
    pn3 = join(['datasets\' dataset(d) '\results\disp\']);
	pn3 = replace(pn3, ' ', '');
    pn4 = join(['datasets\' dataset(d) '\results\results\']);
    pn4 = replace(pn4, ' ', '');
    
    path = replace(join([pn2 '*.jpg']), ' ', '');
    
    imagefiles = dir(path);

    di = 100000000;
    red = [];
    green = [];
    blue = [];
    labels = [];
    anomalyIdx = -1;

    numRegions = numClusters(d);

    nfiles = length(imagefiles);    
    k = 0;
    for ii=1:1:nfiles
        fn = imagefiles(ii).name;
        path2 = replace(join([pn2 fn]), ' ', '');
        real=imread(path2);
        
        path2 = replace(join([pn fn]), ' ', '');
        fake = imread(path2);

        dis = 255*(real-fake);

        dis = 255-dis;
        old = dis;

        [r,c, ch] = size(dis);

        if dataset(d) == "sixray"
            
            [L,Centers] = imsegkmeans(dis,numRegions);
            J = label2rgb(L,im2double(Centers));

            co = color{d};
            for l = 1:length(Centers(:,1))
                sm = (double(Centers(l,1)) - co(1)).^2 + (double(Centers(l,2)) - co(2)).^2 + (double(Centers(l,3)) - co(3)).^2;
                di2 = sqrt(double(sm));

                if di2 < di
                    anomalyIdx = l;
                    di = di2;
                end
            end
            [r,c,ch] = size(dis);
            mask = logical(zeros(r,c));
            mask(L == anomalyIdx) = 1;

            mask = bwmorph(mask,'close');
            mask = bwmorph(mask,'close');
            mask = bwmorph(mask,'close');
            
            mask = bwareaopen(mask, 10000);
            
            t = 10;
            for i = 224:224:c
                mask2(:, i-t:i+t) = 255;
                mask2 = mask2';
                mask2(:, i-t:i+t) = 255;
            end
            
            mask2 = logical(mask2(1:r,1:c));    
            mask3 = mask.*mask2;
            mask = bwareaopen(mask,10000) - mask3;

            mask = bwmorph(mask,'thicken');
            mask = bwmorph(mask,'thicken');
            mask = imdilate(mask,strel('disk',5));
            mask = imdilate(mask,strel('disk',5));
            mask = imclose(mask,strel('disk',5));
            mask = imclose(mask,strel('disk',5));
            
            mask = imfill(mask,'holes');
            
            
            mask = bwareaopen(mask, 40000);
        
        elseif dataset(d) == "gdxray"
            
            [L,Centers] = imsegkmeans(dis,numRegions);
            J = label2rgb(L,im2double(Centers));

            co = color{d};
            for l = 1:length(Centers(:,1))
                sm = (double(Centers(l,1)) - co(1)).^2 + (double(Centers(l,2)) - co(2)).^2 + (double(Centers(l,3)) - co(3)).^2;
                di2 = sqrt(double(sm));

                if di2 < di
                    anomalyIdx = l;
                    di = di2;
                end
            end
            [r,c,ch] = size(dis);
            mask = logical(zeros(r,c));
            mask(L == anomalyIdx) = 1;
            
            mask = bwareaopen(mask, 10000);
            mask = bwareaopen(mask,20000);
            
            mask = bwmorph(mask,'thicken');
            mask = imopen(mask,strel('disk',5));
            mask = imclose(mask,strel('disk',5));
            mask = bwmorph(mask,'thicken');
            mask = bwmorph(mask,'thicken');
            mask = bwmorph(mask,'thicken');
            mask = imclose(mask,strel('disk',5));
            
            mask = imfill(mask,'holes');
            mask = bwareaopen(mask,40000);

        elseif dataset(d) == "compass"
            
            t = 300;
            dis(:,1:t,:) = 0;
            dis(:,end-t:end,:) = 0;
            
            t = 10;
            
            co = color{d};
            [r,c,ch] = size(dis);
            
            for i = 1:r
                for j = 1:c
                    distance = (double(dis(i,j,1)) - co(1)).^2 + (double(dis(i,j,2)) - co(2)).^2 + (double(dis(i,j,3)) - co(3)).^2; 
                    distance = sqrt(distance);
                    
                    if distance < t%abs(dis(i,j,1) - co(1)) > t && abs(dis(i,j,2) - co(2)) > t && abs(dis(i,j,3) - co(3)) > t
                        dis(i,j,1) = 255;
                        dis(i,j,2) = 255;
                        dis(i,j,3) = 255;
                    end
                end
            end
            
            [L,Centers] = imsegkmeans(dis,numRegions);
            J = label2rgb(L,im2double(Centers));

            co = [165 255 230];
            
            for l = 1:length(Centers(:,1))
                sm = (double(Centers(l,1)) - co(1)).^2 + (double(Centers(l,2)) - co(2)).^2 + (double(Centers(l,3)) - co(3)).^2;
                di2 = sqrt(double(sm));

                if di2 < di
                    anomalyIdx = l;
                    di = di2;
                end
            end
            [r,c,ch] = size(dis);
            mask = logical(zeros(r,c));
            mask(L == anomalyIdx) = 1;
            
            mask = bwmorph(mask,'thicken');
            mask = bwmorph(mask,'thicken');
            mask = bwmorph(mask,'thicken');
            mask = imclose(mask,strel('disk',5));
            mask = imclose(mask,strel('disk',5));
            
            mask = bwareaopen(mask,15000);
            mask = imfill(mask,'holes');

        elseif dataset(d) == "opixray"
            
            t = 300;
            dis(:,1:t,:) = 0;
            dis(:,end-t:end,:) = 0;
            
            t = 10;
            
            co = color{d};
            [r,c,ch] = size(dis);
            
            for i = 1:r
                for j = 1:c
                    distance = (double(dis(i,j,1)) - co(1)).^2 + (double(dis(i,j,2)) - co(2)).^2 + (double(dis(i,j,3)) - co(3)).^2; 
                    distance = sqrt(distance);
                    
                    if distance < t%abs(dis(i,j,1) - co(1)) > t && abs(dis(i,j,2) - co(2)) > t && abs(dis(i,j,3) - co(3)) > t
                        dis(i,j,1) = 255;
                        dis(i,j,2) = 255;
                        dis(i,j,3) = 255;
                    end
                end
            end
            
            [L,Centers] = imsegkmeans(dis,numRegions);
            J = label2rgb(L,im2double(Centers));

            co = [198 216 255];
            
            for l = 1:length(Centers(:,1))
                sm = (double(Centers(l,1)) - co(1)).^2 + (double(Centers(l,2)) - co(2)).^2 + (double(Centers(l,3)) - co(3)).^2;
                di2 = sqrt(double(sm));

                if di2 < di
                    anomalyIdx = l;
                    di = di2;
                end
            end
            [r,c,ch] = size(dis);
            mask = logical(zeros(r,c));
            mask(L == anomalyIdx) = 1;
            
            mask = bwareaopen(mask,15000);

            mask = bwmorph(mask,'thin');
            mask = imopen(mask,strel('disk',3));
            
            mask = bwmorph(mask,'thicken');
            mask = bwmorph(mask,'thicken');
            mask = bwmorph(mask,'thicken');
            
            mask = imclose(mask,strel('disk',5));
            mask = imclose(mask,strel('disk',5));
            
            mask = bwareaopen(mask,5000);
            
            mask = imfill(mask,'holes');

        end


        cc = bwconncomp(mask); 
        bboxes = regionprops(cc,'BoundingBox');

        i3 = real;

        for j = 1:length(bboxes)
            patch = imcrop(real, bboxes(j).BoundingBox); 
            patch = imresize(patch, [224 224],'bilinear');

            i3 = insertObjectAnnotation(i3, 'rectangle', bboxes(j).BoundingBox, 'Anomaly','LineWidth',30,'FontSize',72);

            a = i3(:,:,1);
            a(mask == 1) = 255;
            i3(:,:,1) = a;

            k = k + 1;
        end

        h = imshowpair(real, i3, 'montage');

        imwrite(h.CData,replace(join([pn4 fn]), ' ', ''),'JPEG');
        
    end

end