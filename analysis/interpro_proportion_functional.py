#get gene2transcript from gtf 
gtf = '' #gtf File e.g. "09092025_834_Isoforms.filtered.clean.gtf"
gene2transcript = {}
mtDic = {}


with open(gtf,'r') as f:
    for line in f:
        sp = line.strip().split('\t')
        transcript = sp[-1].split(';')[0].split(' ')[-1].strip('\"')
        gene = sp[-1].split(';')[1].split(' ')[-1].strip('\"')
        if sp[2] == 'transcript':
            if gene not in gene2transcript:
                gene2transcript[gene] = []
            gene2transcript[gene].append(transcript)
            
            
            #MT region
            if sp[0] == "scaffold_2" and int(sp[3]) > 49808 and int(sp[4]) < 1730591:
                mtDic[gene] = True
            else:
                mtDic[gene] = False

                
#read interproscan output
interproScan = '' #interproscan output e.g. "isoforms_corrected.faa.tsv"
transcriptDicGo = {}
transcriptDicDomain = {}

with open(interproScan,'r') as f:
    for line in f:
        sp = line.strip().split('\t')
        #domain = sp[4]
        domain = sp[11]

        transcript = sp[0]
        
        if '|' in sp[-1]:
            go = sp[-1].split('|')
        else:
            go = [sp[-1]]
            
            
        
        if transcript not in transcriptDicGo:
            transcriptDicGo[transcript] = []
        transcriptDicGo[transcript] = transcriptDicGo[transcript] + go
            
        if transcript not in transcriptDicDomain:
            transcriptDicDomain[transcript] = []
        transcriptDicDomain[transcript].append(domain)
        
        
geneTotalGo = {}
geneTotalDomain = {}


#X = threshold for proportion of total terms/domains for a transcript to be considered functional
x = .9

domain80Dic = {}
goSameDic = {}
for gene in gene2transcript:
    pass80Go = 0
    pass80Domain = 0
    total = 0
    geneTotalGo[gene] = []
    geneTotalDomain[gene] = []
    for transcript in gene2transcript[gene]:
        if transcript in transcriptDicGo:
            
            geneTotalGo[gene] = geneTotalGo[gene] +  transcriptDicGo[transcript]
            geneTotalDomain[gene] = geneTotalDomain[gene] + transcriptDicDomain[transcript]

    totalGoSet = set(geneTotalGo[gene])
    totalDomainSet = set(geneTotalDomain[gene])
    
    for transcript in gene2transcript[gene]:
        if transcript in transcriptDicGo:

            transcriptSet =  set(transcriptDicDomain[transcript])
            common_elements = transcriptSet.intersection(totalDomainSet)
            percentage = (len(common_elements) / len(totalDomainSet)) 
            if percentage > x:
                pass80Domain += 1

            transcriptSet =  set(transcriptDicGo[transcript])
            common_elements = transcriptSet.intersection(totalGoSet)
            percentage = (len(common_elements) / len(totalGoSet)) 
            if percentage > x:
                pass80Go += 1
            
            total +=1
    
            
    if total != 0:
        domain80Dic[gene] =  pass80Domain/total
        goSameDic[gene] = pass80Go/total
        


        
mtGo = []
nonmtGo = []
mtDomain = []
nonmtDomain = []

finalPlottingDic = {'source':[],'{0}% protein domains'.format(str(x*100)):[], '{0}% gene ontology terms'.format(str(x*100)):[] }
for gene in mtDic:
    if gene in goSameDic:
        if mtDic[gene] == True:
            finalPlottingDic['source'].append('MT')
            finalPlottingDic['{0}% protein domains'.format(str(x*100))].append(domain80Dic[gene])
            finalPlottingDic['{0}% gene ontology terms'.format(str(x*100))].append(goSameDic[gene])
            mtGo.append(goSameDic[gene])
            mtDomain.append(domain80Dic[gene])
        else:
            finalPlottingDic['source'].append('rest of genome')
            finalPlottingDic['{0}% protein domains'.format(str(x*100))].append(domain80Dic[gene])
            finalPlottingDic['{0}% gene ontology terms'.format(str(x*100))].append(domain80Dic[gene])
            nonmtGo.append(goSameDic[gene])
            nonmtDomain.append(domain80Dic[gene])

            
            
print("proportion of transcripts containing {0}% of total functional terms\n".format(str(x*100)),"MT:", statistics.mean(mtGo), "\nrest of genome:",statistics.mean(nonmtGo))


print("proportion of transcripts containing {0}% of total protein domains\n".format(str(x*100)), "MT:",statistics.mean(mtDomain), "\nrest of genome:",statistics.mean(nonmtDomain))
