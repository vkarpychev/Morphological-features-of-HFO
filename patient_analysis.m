%% ========================================================================
% For each channel, it allows to obtain the values of the
% metrics from HFOobj structure - amplitudes,durations,frequencies of the 
% Ripples/FRs, the threshold of the Ripples as well.
% inputs: HFOobj - a structure with the detected HFO events;
% rate - a structure with averaged rate of ripples, FR, and RFR for each channel;
% i - number of patient 
% output: metric - a table with extracted features of the HFO events for each patient.

function metric = patient_analysis(HFOobj,rate,i)
    
    metric.duration = [];
    
    metric.ripple.rate = [];
    
    metric.fr.rate = [];
    
    metric.rfr.rate = [];

    metric.ripple.amplitude = [];
    
    metric.fr.amplitude = [];
            
    metric.ripple.frequency = [];
    
    metric.fr.frequency = [];
    
    metric.ripple.threshold = [];
        
    metric.fn = length(HFOobj(1).result.time)/(5*60)/2;
    
    % the analysis at the level of channels
    for channel = 1:length(HFOobj)
        
        st = HFOobj(channel).result.autoSta(HFOobj(channel).result.mark == 3);
        fin = HFOobj(channel).result.autoEnd(HFOobj(channel).result.mark == 3);
            
        try
        
            for j = 1:length(HFOobj(channel).result.mark(HFOobj(channel).result.mark == 3))
                   
                metric.st = st(j);
                metric.fin = fin(j);
                metric.rfr.rate = [metric.rfr.rate;[rate.rfr(channel) HFOobj(channel).label]];
                metric.ripple.rate = [metric.ripple.rate;[rate.ripple(channel) HFOobj(channel).label]];
                metric.fr.rate = [metric.fr.rate;[rate.fr(channel) HFOobj(channel).label]];
                
                metric = event_analysis(HFOobj(channel),metric);
                                        
            end
        
        catch
            
            continue
            
        end
        
        clear st fin j
        
    end

    % for each channel, it removes the excluded channels and their metrics
    excluded_channel = exclusion_channel(i);
    
    metric.rfr.rate(find(ismember(cat(1,metric.rfr.rate(:,2)),excluded_channel))',:) = [];
    
    metric.ripple.rate(find(ismember(cat(1,metric.ripple.rate(:,2)),excluded_channel))',:) = [];
    
    metric.fr.rate(find(ismember(cat(1,metric.fr.rate(:,2)),excluded_channel))',:) = [];
        
    metric.ripple.amplitude(find(ismember(cat(1,metric.ripple.amplitude(:,2)),excluded_channel))',:) = [];
    
    metric.fr.amplitude(find(ismember(cat(1,metric.fr.amplitude(:,2)),excluded_channel))',:) = [];
    
    metric.duration(find(ismember(cat(1,metric.duration(:,2)),excluded_channel))',:) = [];
    
    metric.ripple.frequency(find(ismember(cat(1,metric.ripple.frequency(:,2)),excluded_channel))',:) = [];
    
    metric.fr.frequency(find(ismember(cat(1,metric.fr.frequency(:,2)),excluded_channel))',:) = [];
    
    metric.ripple.threshold(find(ismember(cat(1,metric.ripple.threshold(:,2)),excluded_channel))',:) = [];
    
    metric = rmfield(metric,{'fn','st','fin'});
    
end

%% ========================================================================
% For each event, it allows to obtain the metrics for the further analysis

function metric = event_analysis(HFOobj,metric)
        
        % for each event, the duration is calculated
        [~,n1] = min(abs(HFOobj.result.time - metric.st));
        [~,n2] = min(abs(HFOobj.result.time - metric.fin));
        
        % for the frequencies, the fft-result -- > the averiging the
        % frequencies of the peaks on the power spectrum plot
                
        % Ripple's part
        metric.duration = [metric.duration;[abs(metric.st - metric.fin) HFOobj.label]];
                
        metric.ripple.amplitude = [metric.ripple.amplitude;[num2cell(abs(min(HFOobj.result.signalFilt(n1:n2))) + ...
                                    abs(max(HFOobj.result.signalFilt(n1:n2)))) HFOobj.label]];
        
        FTs = fft(HFOobj.result.signalFilt(n1:n2)-mean(HFOobj.result.signalFilt(n1:n2)))/...
                                                        length(HFOobj.result.signalFilt(n1:n2));
           
        Fv = linspace(0, 1, fix(length(HFOobj.result.signalFilt(n1:n2))/2+1))*metric.fn;
        
        Iv = find(Fv >= 80 & Fv <= 250);

        [~,frq] = findpeaks(abs(FTs(Iv))*2,Fv(Fv >= 80 & Fv <= 250),'MinPeakHeight',0.05);
        
        metric.ripple.frequency = [metric.ripple.frequency;[mean(frq) HFOobj.label]];
        
        metric.ripple.threshold = [metric.ripple.threshold;[HFOobj.result.THRFR HFOobj.label]];
        
        % FR's part
                
        metric.fr.amplitude = [metric.fr.amplitude;[num2cell(abs(min(HFOobj.result.signalFiltFR(n1:n2))) + ...
                                abs(max(HFOobj.result.signalFiltFR(n1:n2)))) HFOobj.label]];
        
        FTs = fft(HFOobj.result.signalFiltFR(n1:n2)-mean(HFOobj.result.signalFiltFR(n1:n2)))/...
                                                        length(HFOobj.result.signalFiltFR(n1:n2));
        
        Fv = linspace(0, 1, fix(length(HFOobj.result.signalFiltFR(n1:n2))/2+1))*metric.fn; 
        
        Iv = find(Fv >= 250 & Fv <= 500);

        [~,frq] = findpeaks(abs(FTs(Iv))*2,Fv(Fv >= 250 & Fv <= 500),'MinPeakHeight',0.01);
        
        metric.fr.frequency = [metric.fr.frequency;[mean(frq) HFOobj.label]];

end

%% ========================================================================
% For each patient, there is a defined set of the excluded channels based
% on the previous analysis

function excluded_channel = exclusion_channel(i)
            
    if i == 1
        
        excluded_channel = {'8LF1-2';'9LF3-4';'10LF1-2';'10LF2-3';'10LF3-4';'10LF4-5'};
        
    elseif i == 2
        
        excluded_channel = {'6RH4-5';'6RH5-6';'6RH6-7';'6RH7-8';'6RH8-9';'7RH1-2'};
        
    elseif i == 3
        
        excluded_channel = {'1RH1-2';'1RH4-5';'1RH5-6';'1RH6-7';'1RH7-8';'1RH8-9';...
                            '2RH1-2';'2RH2-3';'2RH3-4';'2RH4-5';'2RH5-6';'2RH6-7';'2RH7-8';'6LH4-5'};
    elseif i == 4
        
        excluded_channel = {'1T1-2';'1T2-3';'1T3-4';'1T6-7';'1T7-8';'1T8-9';'1T9-10';'1T10-11';'2T2-3';'2T3-4';'2T4-5';...
                            '2T5-6';'2T6-7';'2T7-8';'3T2-3';'3T3-4';'3T4-5';'3T5-6';'3T6-7';'3T7-8'};
        
    elseif i == 5
        
        excluded_channel = {'1RH3-4';'1RH4-5';'1RH5-6';'1RH6-7';'1RH7-8';'1RH8-9';'2RH1-2';'2RH2-3'};
        
    elseif i == 6
        
        excluded_channel = {'1H1-2';'1H2-3';'1H3-4';'1H4-5';'1H5-6';'1H6-7';'1H7-8';...
                            '2H1-2';'2H2-3';'2H3-4';'2H4-5';'2H5-6';'2H6-7';'2H7-8';'2H8-9';'2H9-10';...
                            '3H1-2';'3H2-3';'3H8-9';'3H9-10';'5I1-2';'5I2-3';'5I3-4';'5I4-5';'5I5-6';'14LF2-3';'14LF3-4';};
        
    elseif i == 7
        
        excluded_channel = {'3T1-2';'3T2-3';'5H5-6';'5H6-7';'5H7-8';'7F3-4';'7F4-5';'8F5-6';'8F6-7'};
        
    elseif i == 8
        
         excluded_channel = {'7O8-9';'9F9-10';'12H7-8';'13H1-2';'14I7-8'};
         
    elseif i == 9
        
         excluded_channel = {'3F4-5';'3F5-6';'3F7-8';'12I5-6';'14I6-7';'12I6-7';'15I2-3';'15I5-6';...
                             '7H1-2';'7H2-3';'7H3-4';'7H4-5';'7H5-6';'7H6-7';'7H7-8';'7H8-9';'8H2-3';'8H3-4';'8H4-5'};
         
    elseif i == 10
        
        excluded_channel = {'1H2-3';'1H4-5';'1H5-6';'1H6-7';'1H7-8';'1H8-9';'1H9-10';'2A1-2';'2A2-3';...
                            '2A4-5';'2A5-6';'2A6-7';'2A7-8';'2A8-9';'2A9-10';'3A1-2';'3A2-3'};
         
    elseif i == 11
        
        excluded_channel = {'1Hip1-2';'1Hip4-5';'1Hip5-6';'1Hip6-7';'2Hip1-2';'2Hip2-3';'2Hip3-4';'5Bins5-6';'6FIns4-5'};
         
    elseif i == 12
        
        excluded_channel = {'7FR9-10';'8IR5-6';'9IR5-6';'10PR1-2';'10PR2-3';'10PR3-4';'16ZR1-2';'16ZR2-3';'16ZR3-4';'16ZR4-5';...
                            '16ZR5-6';'17ZR1-2';'17ZR2-3';'17ZR3-4';'17ZR4-5'};
         
    elseif i == 13
        
         excluded_channel = {'1PA1-2';'6GC9-10';'7HT6-7';'7HT7-8';'7HT1-2';'7HT2-3';'7HT3-4';'7HT4-5';'8HC1-2';'8HC2-3';'8HC3-4';'9TP2-3'};
         
    elseif i == 14
        
        excluded_channel = {'1TT1-2';'5CC8-9';'5CC9-10';'7TH8-9';'7TH9-10';'8TH2-3';'8TH3-4';'1TT3-4';'1TT4-5';'1TT5-6';...
                            '2TA2-3';'2TA3-4';'2TA4-5';'2TA5-6';'2TA6-7';'2TA7-8';'3TH4-5';'3TH5-6';'3TH6-7';'3TH7-8';'3TH8-9'};
        
    elseif i == 15
        
         excluded_channel = {'8GL2-3';'8GL3-4';'2HC2-3';'2HC3-4';'3TB1-2';'3TB2-3';'4GPH8-9'};
         
    elseif i == 16
        
         excluded_channel = {'7TB6-7';'7TB7-8';'8NA2-3';'11HC3-4';'5HT2-3';'5HT3-4';'6HC3-4';'6HC4-5';'6HC5-6';'7TB3-4';'7TB4-5';'8NA3-4';'8NA4-5'};
        
    elseif i == 17
        
         excluded_channel = {'4TT2-3';'4TT3-4';'6PI9-10';'8FT2-3';'8FT3-4';'9RR1-2';'2BB2-3';'2BB3-4';'2BB4-5';...
                             '3CC1-2';'4TT2-3';'4TT3-4';'4TT4-5';'4TT5-6';'4TT6-7'};
    end

end
