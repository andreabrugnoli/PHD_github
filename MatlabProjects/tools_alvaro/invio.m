function [Mi] = invio(M,ind)
% Mi=invio(M,IND) inverses the input/output channel IND 
% in the system M.
%     * M can be a state-space or a transfer model,
%     * Mi is a state-space model,
%     * IND can be a vector of indices,
%     * M and Mi have the same numbers of inputs and outputs.

if size(ind,1)>1, ind=ind';end
[A,B,C,D]=ssdata(M);
ind1=setdiff([1:size(B,2)],ind);
ind2=ind;
B1=B(:,ind1);B2=B(:,ind2);
C1=C(ind1,:);C2=C(ind2,:);
D11=D(ind1,ind1);D12=D(ind1,ind2);D21=D(ind2,ind1);D22=D(ind2,ind2);
if cond(D22)==Inf, 
    disp('The specified channel is not invertible.');
    Mi=[];
    return
else
    D22i=inv(D22);
    Mi=ss(A-B2*D22i*C2,[B1-B2*D22i*D21 B2*D22i],[C1-D12*D22i*C2;-D22i*C2],...
        [D11-D12*D22i*D21 D12*D22i;-D22i*D21 D22i]);
    [dum,ii]=sort([ind1,ind2]);
    Mi=Mi(ii,ii);
end


