# Owner Local EP Phase 4 Drop Findings

- Phase 4 drop sweep on impl=ep_ht, overlap=0.25, prompt=reasoning_brief.
- num_repeats=3, warmup_cases=1. mean ± stdev across reps in summary_agg.csv.
- Best (by mean score desc, mean e2e asc): load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance / hotspot_relief @ rate=0.2 score_mean=0.9453125 e2e_mean=131.69097830587998

|baseline_key|drop_policy|drop_rate|n_reps|score_mean|score_std|e2e_s_mean|e2e_s_std|prefill_tok_s_mean|decode_tok_s_mean|drop_fraction_total_mean|drop_fraction_remote_mean|dropped_weight_mass_fraction_mean|pass|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|round_robin__numa_local_first__min_communication|none|0.0|3|0.93359375|0.010334966058846057|111.08870414396127|1.9842732840847606|238.78299261700556|26.488548067393328|0.0|0.0|0.0|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.05|3|0.93359375|0.010334966058846057|131.72118632805845|7.909630783164629|209.94407958401916|22.24622405847543|0.048434630087541376|0.06471798395345099|0.02254229738056231|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.1|3|0.91796875|0.0|134.04259931389242|0.0|210.21328222888158|22.01008102404417|0.09738989081094344|0.1301385682575059|0.05028273324643925|True|
|round_robin__numa_local_first__min_communication|tail_weight|0.2|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|per_expert_uniform|0.05|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|per_expert_uniform|0.1|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|per_expert_uniform|0.2|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|hot_expert_relief|0.05|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|hot_expert_relief|0.1|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|hot_expert_relief|0.2|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|per_expert_tailtoken|0.05|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|per_expert_tailtoken|0.1|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|per_expert_tailtoken|0.2|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.05|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.1|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|cross_numa_first|0.2|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|random|0.05|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|random|0.1|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|random|0.2|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|hotspot_relief|0.05|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|hotspot_relief|0.1|3|None|None|None|None|None|None|None|None|None|False|
|round_robin__numa_local_first__min_communication|hotspot_relief|0.2|3|None|None|None|None|None|None|None|None|None|False|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|none|0.0|3|0.93359375|0.010334966058846057|113.0980265866965|1.222008237075477|241.9360581704676|25.74495975212979|0.0|0.0|0.0|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|tail_weight|0.05|3|0.9375|0.01790068630842125|138.93764388250807|4.153557659526309|215.64418423854465|21.041600040115735|0.04832917866170073|0.06452566918059378|0.02248638076702919|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|tail_weight|0.1|3|0.9348958333333334|0.005966895436140417|134.37063312763348|4.440132377534808|211.44789003290057|21.709120588359998|0.09718091493734561|0.12974860271811628|0.050143676691472794|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|tail_weight|0.2|3|0.9348958333333334|0.012556836928376244|139.34659917078292|7.7734929029219|214.45435249452532|20.849416782686685|0.20159937328721775|0.26920598143726177|0.11896616748206677|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|per_expert_uniform|0.05|3|0.9361979166666666|0.004510548978043951|130.3388447118923|4.239599095543461|212.31605299618283|21.92469867129846|0.012450027780904822|0.01662362120302364|0.0057941813941381854|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|per_expert_uniform|0.1|3|0.9361979166666666|0.005966895436140417|129.80439294409007|1.2928258116764226|213.90460287655415|22.278518812362766|0.03323703079593965|0.04437812019248696|0.017705030817492817|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|per_expert_uniform|0.2|3|0.9401041666666666|0.011933790872280834|141.28298228311664|9.98528271700963|215.20961243437782|21.17539100213636|0.10032050162884447|0.13393533760185747|0.06390301334443686|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|hot_expert_relief|0.05|3|0.9361979166666666|0.00813150781041458|138.18597852236903|8.989286027692867|206.65089551131814|21.215699051156214|0.025286757445116437|0.03375804725261592|0.014807583156048276|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|hot_expert_relief|0.1|3|0.93359375|0.00390625|133.71545808125907|7.918612800331807|208.05941661757342|21.758975844833017|0.06485178363923383|0.08660994618482369|0.043033343515085115|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|hot_expert_relief|0.2|3|0.9127604166666666|0.013718299157360336|137.58750395212942|3.268055508508672|222.33952030034584|21.21289968088736|0.1410277823887812|0.1882533299769|0.1079577877212186|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|per_expert_tailtoken|0.05|3|0.9440104166666666|0.005966895436140417|128.9012011475861|0.7180190981116442|213.92236038652482|22.445210523642007|0.012481952810633962|0.016667340447111104|0.005809837409680316|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|per_expert_tailtoken|0.1|3|0.9348958333333334|0.00813150781041458|133.48146252520382|5.395348232582675|207.22137080911384|22.170245937851462|0.03333691730602702|0.04451337892073943|0.017765408626478487|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|per_expert_tailtoken|0.2|3|0.9375|0.0067658234670659265|137.69445517209047|1.4906198512542177|210.65666743621054|21.987445712368938|0.10038466177783205|0.13400716078940442|0.06388960485418986|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|cross_numa_first|0.05|3|0.9375|0.010334966058846057|134.84801125510907|7.822403946730286|206.50188009322292|21.366194127127045|0.048411070617464046|0.06463301082926107|0.024495920524562717|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|cross_numa_first|0.1|3|0.9309895833333334|0.018460217290049253|133.4073499245569|0.9766775597889215|209.83966259488966|21.836183497123827|0.0971881669697507|0.1297638726726745|0.055297101599245925|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|cross_numa_first|0.2|3|0.9375|0.0078125|134.82319803054756|4.406474470165303|216.24205867859382|23.03351289735973|0.20164405026695356|0.2692265331730291|0.13626250544179705|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|random|0.05|3|0.9010416666666666|0.023867581744561668|131.87467744015157|2.408223517822214|211.68210482018534|21.07546929170445|0.04843621264700421|0.06466945661420766|0.04777908785683719|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|random|0.1|3|0.8216145833333334|0.059284127280376096|135.47618131696558|11.155815787640591|223.04919502052175|22.514850551862732|0.09727813541768637|0.12988176869225976|0.09593526221782737|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|random|0.2|3|0.6979166666666666|0.0735302494973724|136.7190954014659|5.376663340054208|212.63343012377888|22.328259016040704|0.20161885264859106|0.26921113364972676|0.19888008294610982|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|hotspot_relief|0.05|3|0.9296875|0.0067658234670659265|134.39095300560197|5.04977171197234|222.9646347294606|21.19747302159541|0.00452701504238582|0.006044171379026751|0.0024887344975353712|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|hotspot_relief|0.1|3|0.9309895833333334|0.004510548978043951|128.2017876210933|1.2381219447233138|211.0663859968977|22.732068498589186|0.010322300793875293|0.013781009151688346|0.006123971000928406|True|
|load_balanced_greedy_with_locality_tiebreak__numa_local_first__greedy_balance|hotspot_relief|0.2|3|0.9453125|0.0078125|131.69097830587998|2.489518247140835|210.67346693243167|22.002902456796665|0.020322137200406983|0.027134812879625755|0.013613750198320394|True|
