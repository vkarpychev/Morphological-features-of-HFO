%% ========================================================================
% For each patient, divides the metrics of the channels into several
% regions - HP,pHP,FL,TL,PL,OL. Also, the function removes the other areas
% from the analysis.
% inputs: resultsdir contains patients' folders and a file with the channels' localization;
% folder - a patient's folder with extracted features (xlsx);
% i - number of a patient;
% outputs: metric - a table with the features of the HFO events for all patients;
% vector - a table with median value of the features across channels


function [metric,vector] = region_analysis(resultsdir,folder,i)
        
    restoredefaultpath
    rehash toolboxcache

    file = dir([resultsdir,folder(i).name,'/p*_metric.xlsx']);
    event = readtable([file(1).folder,'/',file(1).name]);
    event = table2cell(event);

    loc = readtable([resultsdir,'/iEEG_localization.xlsx'],'Sheet',folder(i).name);
    ch_ra = channel_ra(i);
                                                
    for j = 1:size(ch_ra,1)
        
    try
        
        id = event(ismember(cat(1,event(:,1)),ch_ra(j,1)),1:10);
        
        if j == 1
            
           if strcmp(ch_ra(j,2),'FL') == 1 || strcmp(ch_ra(j,2),'TL') == 1 || strcmp(ch_ra(j,2),'PL') || strcmp(ch_ra(j,2),'OL')  == 1
               
                ra = [id repmat(ch_ra(j,[2,3]),size(id,1),1) repmat({'Neocortex'},size(id,1),1)];
           
           elseif strcmp(ch_ra(j,2),'HP') == 1 || strcmp(ch_ra(j,2),'pHP') == 1
               
                ra = [id repmat(ch_ra(j,[2,3]),size(id,1),1) repmat({'MTL'},size(id,1),1)];
           
           end
            
        else
            
            if strcmp(ch_ra(j,2),'FL') == 1 || strcmp(ch_ra(j,2),'TL') == 1 || strcmp(ch_ra(j,2),'PL') || strcmp(ch_ra(j,2),'OL')  == 1
            
                ra = [ra; [id repmat(ch_ra(j,[2,3]),size(id,1),1) repmat({'Neocortex'},size(id,1),1)]];
                
            elseif strcmp(ch_ra(j,2),'HP') == 1 || strcmp(ch_ra(j,2),'pHP') == 1
                
                ra = [ra; [id repmat(ch_ra(j,[2,3]),size(id,1),1) repmat({'MTL'},size(id,1),1)]];
                
            end
            
        end
        
     catch
            
           continue
            
     end
                    
    end
    
    ra = [ra repmat({'RA'},size(ra,1),1)];
          
    electrode = electrode_transform(loc);
   
    ch_nra = setdiff(event(:,1),ch_ra);
    
    label = extractBefore(ch_nra, '-');
    
    [~,a,b] = intersect(electrode,label);
    
    ch_nra = [ch_nra(b,1) electrode(a,2) repmat({'TN'},size(a,1),1)];
    
    [~,reindex] = sort(str2double(regexp(ch_nra(:,1),'\d+','match','once')));
    
    ch_nra = ch_nra(reindex,1:3);
    
    for t = 1:size(ch_nra,1)
        
    try
        
        id = event(ismember(cat(1,event(:,1)),ch_nra(t,1)),1:10);
        
        if t == 1
            
           if strcmp(ch_nra(t,2),'FL') == 1 || strcmp(ch_nra(t,2),'TL') == 1 || strcmp(ch_nra(t,2),'PL') || strcmp(ch_nra(t,2),'OL')  == 1
            
                nra = [id repmat(ch_nra(t,[2,3]),size(id,1),1) repmat({'Neocortex'},size(id,1),1)];
                
           elseif strcmp(ch_nra(t,2),'HP') == 1 || strcmp(ch_nra(t,2),'pHP') == 1
               
                nra = [id repmat(ch_nra(t,[2,3]),size(id,1),1) repmat({'MTL'},size(id,1),1)];
               
           end
           
        else
            
            if strcmp(ch_nra(t,2),'FL') == 1 || strcmp(ch_nra(t,2),'TL') == 1 || strcmp(ch_nra(t,2),'PL') || strcmp(ch_nra(t,2),'OL')  == 1
            
                nra = [nra; [id repmat(ch_nra(t,[2,3]),size(id,1),1) repmat({'Neocortex'},size(id,1),1)]];
                
            elseif strcmp(ch_nra(t,2),'HP') == 1 || strcmp(ch_nra(t,2),'pHP') == 1
                
                nra = [nra; [id repmat(ch_nra(t,[2,3]),size(id,1),1) repmat({'MTL'},size(id,1),1)]];
                
            end
            
        end
        
    catch
            
        continue
            
    end
                    
    end
    
    nra = [nra repmat({'non-RA'},size(nra,1),1)];

    metric = [ra; nra];
    
    index = cellfun(@isnan,metric(:,4),'uni',false);
    
    index = cellfun(@any,index);

    metric(index,:) = [];
    
    vector = vector_transform(metric);
    
end

%% ========================================================================
% For each patient, channels inside the resected HFO areas.
% Each channel has its localization and type (TN, TP, FN, FP) according to seizure outcome

function ch_ra = channel_ra(i)

     if i == 2

        ch_ra = {'3LH2-3','HP','TP';'3LH3-4','HP','TP';'9LF1-2','FL','TN';'9LF2-3','FL','TN';'18RF1-2','FL','TN'};
        
     elseif i == 3
           
        ch_ra = {'6RH1-2','pHP','TN';'6RH2-3','pHP','TN';'6RH3-4','pHP','TN';'8LH1-2','HP','TP';'8LH2-3','HP','TP'};
             
     elseif i == 4
         
         ch_ra = {'1RH2-3','HP','TN';'1RH3-4','HP','TN';'1RH9-10','TL','TN'};
              
     elseif i == 5
         
         ch_ra = {'1T4-5','HP','TN';'1T5-6','HP','TN';'3T1-2','pHP','TN'};
              
     elseif i == 6
         
         ch_ra = {'1RH1-2','HP','TN';'1RH2-3','pHP','TN'};

     elseif i == 9
         
         ch_ra = {'1H8-9','TL','TN';'1H9-10','TL','TN';'12LH1-2','HP','TP'};
         
     elseif i == 10
         
         ch_ra = {'4T1-2','HP','TP';'5H1-2','HP','TP'};
         
     elseif i == 11
         
         ch_ra = {'12H1-2','HP','TN';'12H2-3','HP','TN'};
         
     elseif i == 15
         
         ch_ra = {'8H1-2','HP','TN'};
         
     elseif i == 18
         
         ch_ra = {'2A3-4','HP','TN'};
              
     elseif i == 19
         
          ch_ra = {'1Hip2-3','pHP','TN';'1Hip3-4','TL','TN';'3T2-3','OL','FP';'3T3-4','OL','FP';'3T4-5','OL','FP'};

     elseif i == 20
         
         ch_ra = {'13HL1-2','HP','TP';'13HL3-4','HP','TP';'14HL2-3','HP','TP'};
         
     elseif i == 31
         
         ch_ra = {'8HC4-5','HP','TN'};
         
     elseif i == 34
         
         ch_ra = {'2TA1-2','HP','TN';'3TH2-3','HP','TN';'3TH3-4','HP','TN'};
         
     elseif i == 37
         
         ch_ra = {'2HC1-2','pHP','TN';'2HC4-5','TL','TN';'5HT1-2','HP','TN';'5HT2-3','HP','TN'};
         
     elseif i == 40
         
         ch_ra = {'6HC1-2','HP','TN';'6HC2-3','HP','TN'};
         
     elseif i == 42
         
         ch_ra = {'3CC2-3','HP','TN';'5GG8-9','PL','TP';'5GG9-10','PL','TP'};
         
     end

end

%% ========================================================================
% Extracting the locations of the channesls from localization's file 

function electrode = electrode_transform(loc)
    
    exluded_channel = find(ismember(cat(1,loc{:,8}),'no_label_found'))';
    loc(exluded_channel,:) = [];
    
    electrode(:,1) = loc{:,1};
    electrode(:,2) = extractBefore(loc{:,8}, ',');
    
    % hippocampus 
    HPL = find(contains(loc{:,8},'Hipp')&contains(loc{:,8},'Left'));
    electrode(HPL,2) = {'HP'};
    electrode(HPL,3) = {'L'};
    
    HPR = find(contains(loc{:,8},'Hipp')&contains(loc{:,8},'Right'));
    electrode(HPR,2) = {'HP'};
    electrode(HPR,3) = {'R'};
    
    % parahippocampus 
    pHPL = find((contains(electrode(:,2),'PhG')|contains(electrode(:,2),'FuG'))&contains(loc{:,8},'Left'));
    electrode(pHPL,2) = {'pHP'};
    electrode(pHPL,3) = {'L'};
    
    pHPR = find((contains(electrode(:,2),'PhG')|contains(electrode(:,2),'FuG'))&contains(loc{:,8},'Right'));
    electrode(pHPR,2) = {'pHP'};
    electrode(pHPR,3) = {'R'};
    
    % temporal lobe
    TLL = find(contains(electrode(:,2),'TG')&contains(loc{:,8},'Left'));
    electrode(TLL,2) = {'TL'};
    electrode(TLL,3) = {'L'};
    
    TLR = find(contains(electrode(:,2),'TG')&contains(loc{:,8},'Right'));
    electrode(TLR,2) = {'TL'};
    electrode(TLR,3) = {'R'};
    
    % frontal lobe                                     
    FLL = find((contains(electrode(:,2),'FG')|contains(electrode(:,2),'Or')|...
              contains(electrode(:,2),'PrG')|contains(electrode(:,2),'CG'))&contains(loc{:,8},'Left'));
    electrode(FLL,2) = {'FL'};
    electrode(FLL,3) = {'L'};
    
    FLR = find((contains(electrode(:,2),'FG')|contains(electrode(:,2),'Or')|...
              contains(electrode(:,2),'PrG')|contains(electrode(:,2),'CG'))&contains(loc{:,8},'Right'));
    electrode(FLR,2) = {'FL'};
    electrode(FLR,3) = {'R'};
    
    % parietal lobe
    PLL = find((contains(electrode(:,2),'IPL')|contains(electrode(:,2),'SPL')|...
                   contains(electrode(:,2),'Pcun'))&contains(loc{:,8},'Left'));
    electrode(PLL,2) = {'PL'};
    electrode(PLL,3) = {'L'};
    
    PLR = find((contains(electrode(:,2),'IPL')|contains(electrode(:,2),'SPL')|...
                   contains(electrode(:,2),'Pcun'))&contains(loc{:,8},'Right'));
    electrode(PLR,2) = {'PL'};
    electrode(PLR,3) = {'R'};
    
    % occipital lobe
    OLL = find((contains(electrode(:,2),'OG')|contains(electrode(:,2),'Oc'))&contains(loc{:,8},'Left'));
    electrode(OLL,2) = {'OL'};
    electrode(OLL,3) = {'L'};
    
    OLR = find((contains(electrode(:,2),'OG')|contains(electrode(:,2),'Oc'))&contains(loc{:,8},'Right'));
    electrode(OLR,2) = {'OL'};
    electrode(OLR,3) = {'R'};
    
    exclude = find(contains(electrode(:,2),'INS')|contains(electrode(:,2),'BG')|...
                   contains(electrode(:,2),'Tha')|contains(electrode(:,2),'PoG')|...
                   contains(electrode(:,2),'pSTS')|contains(electrode(:,2),'PCL')|...
                   contains(electrode(:,2),'Amyg'));
               
    electrode(exclude,:) = [];
    
    electrode = electrode(~cellfun('isempty',electrode(:,2)),1:3);

end

%% ========================================================================
% Returns the average values in the channels

function vector = vector_transform(metric)

    channel = unique(metric(:,1));
    
    for i = 1:length(channel)
        
        number = find(ismember(cat(1,metric(:,1)),channel(i)));
        
        if exist('vector','var') == 0
                                                
            vector = [num2cell(median(cell2mat(metric(number,2:10)),1))...
                      metric(number(1),11:14)];
            
        else
            
            vector = [vector; [num2cell(median(cell2mat(metric(number,2:10)),1))...
                               metric(number(1),11:14)]];
        
        end
        
    end
    
    vector = [channel vector];

end
