%ScriptHW

creditDS = readtable('CreditRating_Historical.dat');
%create output matrix
[n,k]=size(creditDS);

x=creditDS(:,[1:7]);
xx=table2cell(creditDS(:,[8]));
y=string(xx);
Rating=zeros(n,1);

%labeling
for i=1:n
    if y(i)=='CCC'
        Rating(i)=-1;
    else Rating(i)=1;
    end
end
zz=[x array2table(Rating)];

zzz = sortrows(zz,'Rating','descend');
w=cell2mat(table2cell(zzz(:,[8])));
a=groupcounts(w);

sz=[430 8];
varNames=["ID","WC_TA","RE_TA","EBIT_DA","MVE_BVTD","S_TA","Industry,Rating"];
varTypes=["double","double","double","double","double","double","double","double"];
T = table('Size',sz,'VariableTypes',varTypes); 

rng(1212);
r = floor(rand(300,1)*3803)+1;
for i=1:300
    for j=1:8
        T(i,[j])=zzz(r(i),[j]);
    end
end

for i=1:130
    for j=1:8
        T(300+i,[j])=zzz(3802+i,[j]);
    end
end
        
