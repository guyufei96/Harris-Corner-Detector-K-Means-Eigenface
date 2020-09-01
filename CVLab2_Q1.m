%%
% CLAB2 Task-1: Harris Corner Detector
% Yufei Gu (Your u65447614)
a = imread('Harris_4.jpg');
bw = rgb2gray(a);
C = corner(bw,'harris');

bw = double(bw);

sigma = 1; thresh = 0.0005; % Parameters, add more if needed
% Derivative masks
dx = [-1 0 1; -1 0 1; -1 0 1];
dy = dx'; % dx is the transpose matrix of dy
% compute x and y derivatives of image
Ix = conv2(bw,dx,'same');
Iy = conv2(bw,dy,'same');
g = fspecial('gaussian',max(1,fix(3*sigma)*2+1),sigma); %generate the Gaussian convolution kernel
Ix2 = conv2(Ix.^2,g,'same'); % x and x
Iy2 = conv2(Iy.^2,g,'same'); % y and y
Ixy = conv2(Ix.*Iy,g,'same'); % x and y

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task: Compute the Harris Cornerness %
[m,n] = size(bw);
R = zeros(m,n); 
k = 0.05;   %usually use between 0.04~0.06
for i = 1:m
    for j = 1:n
        M = [Ix2(i,j) Ixy(i,j) ; Ixy(i,j) Iy2(i,j)]; %calculation of M matrix which computed from image derivatives
        R(i,j) = det(M) - k*(trace(M))^2; %calculate R to judge if it is on the edge
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Task: Perform non-maximum suppression and %
% thresholding, return the N corner points %
% as an Nx2 matrix of x and y coordinates %
Rmax = max(max(R));
t = thresh * Rmax; %Rmax * thresh as a threshold, judge if this point is corner
for i = 1:m
    for j = 1:n
        if R(i,j) < t %compare with threhold
            R(i,j) = 0; %give 0 to the not corner point
        end
    end
end
corner = imregionalmax(R); %identify the regional maxima 
[posr,posc] = find(corner == 1);% find the regional maxima point position




bw = uint8(bw); 

figure;
subplot(1,2,1), imshow(bw), title('Harris'); %show the origin picture
hold on
for i = 1:length(posr)
    plot(posc(i), posr(i), 'r+'); %mark the corner in the picture with '+'
    
end
subplot(1,2,2), imshow(bw), title('inbuild');
hold on
plot(C(:,1),C(:,2),'r+');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%