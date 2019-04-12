% getDataFromNASTRAN.m
% Chiara Toglia - Thales Alenia Space Italia
% edited by Hari Murali - ISAE
%-----------------------------------------------------------------
% PURPOSE
% Read data from Patran *.bdf and Nastran output *.f06 files
%
% SYNOPSIS
%  [M, J, LS, EV, omega, zeta, DPa0, cdP, cdC, flagFatal] = 
%  getDataFromNASTRAN(F06filename,BDFfilename,rfindex,damping,P_gp,C_gp)
%
% INPUT ARGUMENTS
% F06filename           Name of NASTRAN input file with extension .f06
% BDFfilename           Name of PATRAN bdf file with extension .bdf
% rfindex               Index to denote whether the function is invoked for
%                       a rigid or flexible body 'r' --> rigid ,'f' --> flexible        
% damping               Damping ratio - same value for all modes
% P_gp                  Grid point number corresponding to nodal point P (point of 
%                       attachment of appendage and upstream)
% C_gp                  Grid point number corresponding to nodal point C (where the
%                       modal shape is to be extracted)
%
% OUTPUT ARGUMENTS
% M                     Mass [1]
% J                     Inertia matrix at CG [3x3]
% xcg                   Center of gravity of the body [1x3]
% LS                    Matrix of modal participation factors in translation 
%                       and rotation [6 x nFlex]
% omega                 Vector of vibration frequencies [nFlex x 1]
% EV_C                  Eigen vector/Modal shape at point C [6 x nFlex]
% zeta                  Vector of damping ratios [nFlex x 1]
% DPa0                  Residual mass matrix [6x6]
% cdP                   Nodal co-ordinates of point P with respect to GFF
% cdC                   Nodal co-ordiantes of point C with respect to GFF
% flagFatal             Flag to determine whether the NASTRAN run was successful or not [1]


function [M, J, xcg, LS, omega, EV_C, zeta, DPa0, cdP, cdC, flagFatal] = getDataFromNASTRAN(F06filename,BDFfilename,rfindex,damping,P_gp,C_gp,nf)

M = [];
J = [];
L = [];
S = [];

matEV=[];
omega = [];
zeta = [];
xcg = [];
EVT = [];
EVR = [];
mat1 = [];
matNDcd_P = [];
matNDcd_C = [];


flagFatal = 0;
P_str = num2str(P_gp);
C_str = num2str(C_gp);
indMode = 1;
indModalFact = 1;
EVRow=1; 
indEV=1;

% Data manipulation of the NASTRAN output .f06 file

fid = fopen(F06filename,'r');
while ~feof(fid) % While the end of the file has not been reached,
    row = fgetl(fid); %Return the next line of the file as a string to the variable 'row'
    
    % Check there is no FATAL ERROR, if so exit with a flag
    if (strfind(row, 'FATAL') > 0)
        flagFatal = 1;
        break
    end
    %-------------------------------
    % Get inertia parameters
    if strncmp(row,'                                                                M O',67) 
    % line preceeding to mass and inertia matrices
        
    for ijkRow = 1:6 %for each of the six rows,
            row = fgetl(fid); % fill the variable 'row' with the string corresponding 
                              % to the next line
            mat1ch = row(25:107); % Assigns the numerical values in the string 
                      % (i.e. corresponding to the six elements of the row)
            for ijk = 1:6 % for each of the six elements in the selected row
                
                nElem = (ijk-1)*14+1; %cursor index: 
                    % each element takes 12 elements for numbers, 
                    % 1 for sign and 1 element for the blank space after the power  
                
                    mat1(ijkRow,ijk) = str2double(mat1ch(nElem:nElem+12)); 
                    % In matl fill the numbers (converted from strings) along the rows and columns
            end
    end
       
        M = mat1(1);
        xcg = [mat1(2,6) -mat1(1,6) mat1(1,5)]/M;
        asym_xcg = [0 -xcg(3) xcg(2); xcg(3) 0 -xcg(1); -xcg(2) xcg(1) 0]; % = OXCGvec - OOvec
        tau_xcg = [eye(3) asym_xcg; zeros(3) eye(3)] ;
        MO_xcg = tau_xcg'*mat1*tau_xcg;
        J = MO_xcg(4:6,4:6);
    
    end

  
    
    %--------------------------------
    % Get modal frequencies
    if(rfindex == 'f') 
        
    if strncmp(row,'                                              R E A L   E I G E N V A L U E S',77)
        kOmega = 1;
        while (kOmega)
            row = fgetl(fid);     nRow = length(row); %row terminates after the last character in the line; blank spaces after that are not counted
            if (strncmp(row,'                                       M O D A L   E F F E C T I V E   M A S S   S U M M A R Y',94)||indMode==nf)
                kOmega = 0;
            end
            if (nRow > 8)
                if (str2double(row(1:9)) == indMode)
                    omega(indMode,1) = str2num(row(48:60));
                    indMode = indMode + 1;
                end
            end
        end
    end
    % Get modal participation factors
   
    if strncmp(row,'                                                     MODAL PARTICIPATION FACTORS',80)
        kLS = 1;
        while (kLS)
            row = fgetl(fid);     nRow = length(row);
            if (strncmp(row,'                                                         MODAL EFFECTIVE MASS',77) || indModalFact==nf)
                kLS = 0;
            end
            if (nRow > 8)
                if (str2double(row(1:9)) == indModalFact)
                    for ijk = 1:6
                        nElem = 30 + (ijk-1)*18;
                        Gmat(indModalFact,ijk) = str2double(row(nElem:nElem+12));
                    end
                    indModalFact = indModalFact + 1;
                end
            end
        end
    end
    
    %Get Eigen vectors at point C 
   
   if(length(row)>80)
    if strcmp(row(33:80),'         R E A L   E I G E N V E C T O R   N O .')
            kEV = 1;
        while (kEV) %Is it the Eigen vector table? Yes, enter!
            row = fgetl(fid);     nRow = length(row);
            if (indEV == indMode)
             kEV = 0; %No longer in the Eigen vector table
            end
            
            if (nRow > 8) %Not needed
                if (str2num(row(7:14))==C_gp)
                    indEV = indEV + 1;
                    matEVch = row(26:nRow);
                    for EVCol=1:5
                        nElem = (EVCol-1)*15+1;
                        matEV(EVRow,EVCol) = str2double(matEVch(nElem+1:nElem+13));
                    end
                    nElem=76;
                    matEV(EVRow,6) = str2double(matEVch(nElem+1:end));
                    EVRow = EVRow + 1;
                end
            end
         end
    end
   end
    
  end
end
 
   
fclose(fid);


fid = fopen(BDFfilename,'r');
while ~feof(fid) % While the end of the file has not been reached,
    row1 = fgetl(fid); %Return the next line of the file as a string to the variable 'row1'
    % Check there is no FATAL ERROR, if so exit with a flag
    if (strfind(row1, 'FATAL') > 0)
        flagFatal = 1;
        break
    end
    % Get nodal co-ordinates
    if strncmp(row1,'$ Nodes of the Entire Model',27)
            kND = 1;
        while (kND) %Is it the nodal co-ordinates table? Yes, Enter!
            row1 = fgetl(fid);     
            nRow1 = length(row1);
            if (strncmp(row1,'$ Loads for Load Case',21))
             kND= 0; %No longer in the nodal co-ordinates table
            end
           if (nRow1 > 10) 
                if (str2num(row1(10:16)) == P_gp)
                    matNDinit = row1(25:nRow1); %Matrix of nodal co-ordinates
                    for NDcd_col=1:2
                        nElem = (NDcd_col-1)*8+1;
                        matNDcd_P(NDcd_col) = str2double(matNDinit(nElem:nElem+7));
                    end
                    nElem = 17;
                    matNDcd_P(3) = str2double(matNDinit(nElem:end));
                end
            end
            
           if(rfindex == 'f') 
            if (nRow1 > 8) %Not needed
                if (str2num(row1(10:16)) == C_gp)
                    matNDinit = row1(25:nRow1); %Matrix of nodal co-ordinates
                    for NDcd_col=1:2
                        nElem = (NDcd_col-1)*8+1;
                        matNDcd_C(NDcd_col) = str2double(matNDinit(nElem:nElem+7));
                    end
                    nElem = 17;
                    matNDcd_C(3) = str2double(matNDinit(nElem:end));
                end
            end
           else
               matNDcd_C=0;
           end
           
        end
    end
  
                   
end
fclose(fid);

%results
cdP=matNDcd_P;
cdC=matNDcd_C;

if(rfindex == 'f') %% for flexible body,
   L = Gmat(:,1:3);
   S = Gmat(:,4:6);
   EVT = matEV(:,1:3);
   EVR = matEV(:,4:6);
   zeta = damping*ones(length(omega),1);
   LS = [L S];
   EV_C = [EVT EVR]';
   CGPvec=cdP-xcg ;
   tauCGP=[eye(3) -[0 CGPvec(3) -CGPvec(2);
               -CGPvec(3) 0 CGPvec(1);
              CGPvec(2) -CGPvec(1) 0] ; 
              zeros(3) eye(3)];
   DPa0 = tauCGP'*MO_xcg*tauCGP-LS'*LS;
   
%%-----------   
%%---Checks:  
%    MO_xcg
%    LS
%   EV_C
%%-----------    

else  %%rigid case
   omega = 0;
    zeta = 'Inf';
    LS = 0;
    EV_C = 0;
    DPa0 = 'NA';
    
end

end
