#ifndef DATA_H
#define DATA_H

const int M = 32;

const double A[M][M] = {
{36.1388986346293, 0.315388335234042, 0.8927450954350494, 0.8361074980318122, 0.5563992254958807, 0.4043602463863352, 0.2507829682045096, 0.9514385247813856, 0.08016758028914799, 0.7115525333244146, 0.7959895122299397, 0.769773465786834, 0.1147184947927008, 0.8425453393073765, 0.821825405341388, 0.1291853265687589, 0.1988718716017328, 0.5361856283195963, 0.227578104749309, 0.1459643536632206, 0.8104613828653524, 0.7003129217014571, 0.1479302875233983, 0.5747814996469607, 0.4065512047640914, 0.03085293501000401, 0.8588858677431421, 0.3500323823609006, 0.5620858753890819, 0.8001556210970601, 0.1359651181955606, 0.76049575689726},
{0.2684103390460554, 24.96548530336966, 0.9412141471725195, 0.6029398870008171, 0.9022575790989763, 0.5272216626560322, 0.1726433028836512, 0.2411667975547859, 0.10322188389668, 0.4956136136282368, 0.8910721273085345, 0.7085596750111053, 0.8782203676598626, 0.2309315932858777, 0.9236175469316598, 0.718500384324103, 0.5298160073648487, 0.7860108363753791, 0.7932708875354897, 0.1218998637289809, 0.8146614239954466, 0.1302724353678604, 0.04796907812811584, 0.840135018001957, 0.003135496690081107, 0.4021472009416044, 0.3912544548865028, 0.9488301614890352, 0.6515158978839859, 0.4425949769598674, 0.5573263330502223, 0.4688446359827483},
{0.9559544579436829, 0.08423225366910285, 28.73491527391368, 0.357503814886889, 0.001389341540959643, 0.9251909734812354, 0.1367291556372358, 0.2226995181147517, 0.3773809916128418, 0.9228683155117412, 0.8818139724778429, 0.1648180156492425, 0.299923463037133, 0.975079600682372, 0.3026642984483724, 0.6354624919655492, 0.1299636315256266, 0.01569414117595451, 0.9244492570187262, 0.9286131019245735, 0.1970445257664452, 0.1069008334649988, 0.3202919683645359, 0.4418487915999226, 0.7404527192543942, 0.9493679628554766, 0.9522918447458122, 0.8168598533290187, 0.08927012144093043, 0.2932268502481849, 0.9528896802307496, 0.3991925520411644},
{0.5187460850753904, 0.9503074141239792, 0.6150257832683216, 37.88274771801413, 0.5672072519565552, 0.5032505921881187, 0.7258580557095763, 0.4498505261328129, 0.9630538580881438, 0.6462353839883049, 0.4807478260103086, 0.1979277360336338, 0.6292226403323893, 0.9651769148548957, 0.9742127065143399, 0.9106951441853282, 0.7004356251985056, 0.3842248618003862, 0.6580934600110413, 0.8698348072447692, 0.853172754273921, 0.6937815964595412, 0.2550428552142516, 0.5631180226878543, 0.185026913674354, 0.5573866661310491, 0.1836993279931725, 0.9674191918414049, 0.1176685591088289, 0.3534389359149324, 0.4026391582434043, 0.3763290547852403},
{0.4813844619940866, 0.4494732498231049, 0.861156096290118, 0.174109669418762, 33.57564133977083, 0.972900376463036, 0.1553836231577477, 0.7166767380201842, 0.2686164528130865, 0.8670866730017419, 0.4783509661892045, 0.3927979748596165, 0.6778831878748227, 0.0489908121058981, 0.632998144664739, 0.2522825019930593, 0.1212284195709889, 0.3465976601245047, 0.954747644071591, 0.4316710644866827, 0.6709988830966118, 0.1238886335261719, 0.1644949174966715, 0.7796317435579985, 0.2004584171745058, 0.1833486647909641, 0.5254359289586886, 0.9400155844449567, 0.8778349273374919, 0.1696229420309703, 0.4068646812705405, 0.04949429060220793},
{0.38629084713358, 0.3122179856460643, 0.3035335669701725, 0.341887242860399, 0.9556588630659049, 27.7905258885769, 0.2942376950319104, 0.2384718688859943, 0.5440376376985687, 0.6419602662602065, 0.2363626625757455, 0.1681456839896215, 0.6797567375515234, 0.01348043985025642, 0.9355953130574538, 0.918016369991198, 0.3581069214006801, 0.9809978171770457, 0.1644119090548696, 0.2462307461612143, 0.5489821517212365, 0.9986098995022689, 0.2240056650750459, 0.9562466834983795, 0.581543032220886, 0.1734615635326724, 0.9793436960350088, 0.8031658201273085, 0.6848488907937798, 0.05265023568883047, 0.6048968641616358, 0.3234643851171451},
{0.4604277794015528, 0.6125918535605077, 0.6060634635986959, 0.6209932250379916, 0.8550861209282954, 0.7024429853038007, 34.13898118551879, 0.9664210954806808, 0.7820199882058649, 0.393356969782059, 0.01713071956262851, 0.4473883056325592, 0.03215294383858318, 0.9536936002630122, 0.1579885230433963, 0.6569303634353408, 0.1335006711793528, 0.8570634995333524, 0.262294313600523, 0.2777067263574407, 0.1306629722009262, 0.7567919586560845, 0.983046412036246, 0.9489918938701501, 0.2972370937136671, 0.9857017201357137, 0.6294915468802235, 0.0932477442942357, 0.3248608199403519, 0.7915538063826997, 0.2759651038219348, 0.5438905699770445},
{0.6368148998368214, 0.5562393576471737, 0.9630353493260607, 0.2722466543878645, 0.05371434908794141, 0.8729368287990957, 0.7066216600785199, 26.41365820583654, 0.9755789890088614, 0.7384227262985762, 0.9015201289114889, 0.5578220159298785, 0.3955324590908872, 0.7902754321444896, 0.8464909968227186, 0.4412429028962148, 0.3179854558551913, 0.4982762346448417, 0.6002067809569992, 0.9086804883536787, 0.1390613838427833, 0.7206400686127241, 0.9948193838106271, 0.5859192820403588, 0.8995080659874219, 0.4080628949407641, 0.3173767887965515, 0.9297012860763596, 0.2481871956448958, 0.6023987718622, 0.4971220054739419, 0.03362334391162362},
{0.5992601546408357, 0.9798847199147064, 0.4792628604323003, 0.839053300629395, 0.9355816671426738, 0.1187922751443567, 0.7934842581141441, 0.5915715461447573, 31.18195814189157, 0.7436175770667255, 0.1985169871279426, 0.6605102970431734, 0.4396570589322868, 0.3623156874312489, 0.835618095193891, 0.3685430493593846, 0.1273040857733671, 0.8355279391903493, 0.3310783658014559, 0.8143375786211834, 0.9795827669449172, 0.2478401508337486, 0.08540580681348579, 0.4216197892325915, 0.7617504113043634, 0.4421749425883971, 0.5832963030654041, 0.7844435508992321, 0.1753725877431113, 0.7729222342606227, 0.5774473634216976, 0.6533859261693797},
{0.5412779612808457, 0.7794442907556793, 0.1776144143922305, 0.002004984584171687, 0.7513025646055181, 0.9269695123537299, 0.6440071761926597, 0.9851870953743136, 0.9570027425329179, 24.68079249899198, 0.1539917885765881, 0.513810399245326, 0.2116974833920497, 0.9199760494681163, 0.4244682189776069, 0.8163392815060699, 0.8645279034710516, 0.01656781543699553, 0.7740683750464041, 0.1942299549703672, 0.02091713366254537, 0.2763923156081898, 0.04300692038390146, 0.9220563989191833, 0.07769967483105948, 0.849753940328568, 0.2268642414149765, 0.6164091729847846, 0.5652164099577264, 0.4854710855972025, 0.4318934256958488, 0.9838066988150694},
{0.3279612250946837, 0.60546687208926, 0.3668670602822451, 0.6625352701079914, 0.03969700085026533, 0.9573995652167753, 0.1269058946547638, 0.7497583397843299, 0.4790208697145832, 0.8243363263441077, 30.08735091199858, 0.5624868069701431, 0.2280864742971891, 0.5837124515784987, 0.603854938704934, 0.176387365175768, 0.7556576177230795, 0.5003235301209362, 0.09950124959596089, 0.4815676784301058, 0.01663508195047596, 0.6851944235194761, 0.9500733150453351, 0.8651605320236468, 0.107718043087939, 0.5018001071866071, 0.7660091295143301, 0.7775624404140138, 0.4900740126446034, 0.1967639605135901, 0.9878487335002665, 0.2636256748539754},
{0.3242089765150082, 0.1219936680026495, 0.7697374053275318, 0.3223569359037318, 0.7907185785889106, 0.7480855935249449, 0.396604847616181, 0.4591563953014997, 0.7146181195639738, 0.09128817571998886, 0.8853120565512708, 25.31018742778603, 0.7228555834203583, 0.07195205371975903, 0.9813070569415567, 0.8842905193148548, 0.3176411026301143, 0.7103526419564934, 0.4506383089530437, 0.007687800259984323, 0.3219885548112747, 0.7401710172530807, 0.8229100689500334, 0.8708116998661628, 0.160740582571137, 0.9365319357726013, 0.4166676877643471, 0.1845239229008418, 0.1533211775536164, 0.656414435554678, 0.7456425078865029, 0.8044445339293829},
{0.5356531318671226, 0.1416474480324763, 0.4099857768103679, 0.6053989029981625, 0.464890768034587, 0.9393310746704741, 0.2723951002102036, 0.8235278732656187, 0.09169902822262625, 0.1108339880831618, 0.5478562495018738, 0.6798832997404821, 38.05449824779485, 0.9172925763990876, 0.4435601040216984, 0.05965035919381205, 0.07550433399609617, 0.1367399869555553, 0.4727698930188878, 0.4782654818433226, 0.3640300026103366, 0.7862949664446621, 0.5615057821201836, 0.9116677248560096, 0.06542420440717094, 0.5760549816867481, 0.001260311645430101, 0.6843759850899147, 0.2073284583572181, 0.3752236514460011, 0.04183653840724337, 0.7062880539323267},
{0.3463391013605457, 0.5998454778993154, 0.2600763871040702, 0.9702377882924661, 0.5713001085955565, 0.5571523497125113, 0.3638578719730338, 0.9386183359791788, 0.3378102069609375, 0.4104636528742022, 0.4290222279098715, 0.9022930673707998, 0.01273062672599761, 28.88398058547008, 0.1449853295408738, 0.3136212991395953, 0.8232592136239696, 0.6768270654546811, 0.9594860720680891, 0.2741201714143121, 0.8154621669172069, 0.04750294148573324, 0.005951496476091322, 0.4265164308467999, 0.8818494282701919, 0.6335057769952399, 0.1753878359283149, 0.9294917015413322, 0.3609304704876179, 0.5111645558559255, 0.7342130807861283, 0.1295777252727771},
{0.8515027335135958, 0.5919960384901437, 0.06518627422512584, 0.8982194389725079, 0.7628256921305715, 0.2454989749396464, 0.3617916420152557, 0.6373225960428408, 0.7944983642452853, 0.09801372524439826, 0.9932062951234717, 0.3464420926604756, 0.7590072544129401, 0.9780241017730702, 36.59203748444337, 0.1854936423283258, 0.6954673204013597, 0.5649129399174531, 0.2121807454330346, 0.624541060959771, 0.8185530484803711, 0.05918894861006978, 0.9305640399217985, 0.9057743915454312, 0.7109472436297274, 0.2523592099402366, 0.8685139554930524, 0.067495515599916, 0.5849344205064914, 0.5814392766998134, 0.2999963512929472, 0.8704592950158504},
{0.02315942792123509, 0.7084622604254796, 0.6205258495426406, 0.19221898409148, 0.8606424329665696, 0.8120083015332913, 0.07816413218471201, 0.882833965349369, 0.2947627510495757, 0.9457999031408355, 0.2052676264527432, 0.8799119149641963, 0.4299836066712362, 0.8801117325332795, 0.1802868948108384, 32.49181811213781, 0.8740262001785492, 0.5742655064037048, 0.9259232012225695, 0.5265827522821169, 0.9238110442815041, 0.1613136159205809, 0.7645198329600915, 0.6031871743388192, 0.1144816242328566, 0.1657364567981782, 0.4944018960482477, 0.1264935172210589, 0.2909401150876114, 0.3912510618034273, 0.4943316728995698, 0.2276222708671243},
{0.7733267496683659, 0.6318032946583003, 0.3365391775350925, 0.4952030377087892, 0.757124995330547, 0.2977946506882442, 0.3419366713496554, 0.5674294587984398, 0.6185696547925368, 0.6075901741847386, 0.2431202222093472, 0.1149922896054414, 0.59210003558844, 0.3686929197548848, 0.4577753227652613, 0.8456381732648131, 31.86544652204674, 0.03571057312658237, 0.3138621096447392, 0.393637562557571, 0.2079150756296673, 0.7990782087748178, 0.525129192853283, 0.5342722453393282, 0.3508008263379546, 0.3201007004148345, 0.6203396727852216, 0.8549702693689464, 0.1940551866048377, 0.1987267458657356, 0.7900630316836846, 0.9437102799707743},
{0.2784293400954332, 0.175924503467537, 0.3013936432693535, 0.7927385573390431, 0.937609108538401, 0.1086335720636832, 0.8889543169549448, 0.6916576170236215, 0.8463120085806742, 0.2285278429977403, 0.3523021227238772, 0.225470272618167, 0.001713297649014778, 0.9988640471180303, 0.2562429684459585, 0.4817009262186827, 0.8917978348444094, 35.98513294312213, 0.6268767493369318, 0.3478836652185649, 0.8170902981224528, 0.5383579889399733, 0.4507419201237234, 0.7140227774371762, 0.1781510247235863, 0.4325158772659346, 0.8617978322510992, 0.7461656750044886, 0.528921418127813, 0.5022258367599577, 0.9061694072472336, 0.3060705790451543},
{0.1535100115900064, 0.9330845508048428, 0.6456577549536536, 0.8951422749764059, 0.766083110611957, 0.4485701525365883, 0.7714893908093697, 0.6715633311141846, 0.5309587793504916, 0.4726249474352174, 0.08128006434907802, 0.5440885834904828, 0.9213602138048442, 0.4379594366376358, 0.9152724762878421, 0.4303022121372314, 0.3235560108841075, 0.4787468160030949, 29.72021661739715, 0.8887705180527131, 0.9418834032297098, 0.9471011216669549, 0.5316289815541503, 0.5848114309461203, 0.9373943736315323, 0.3445527559346785, 0.4894575374633635, 0.9819346717531128, 0.2289949149280231, 0.8427826480751647, 0.4330601277944722, 0.1010224543298658},
{0.8286224626872737, 0.1645952239189524, 0.06587656199306906, 0.8345684503413416, 0.9417759092632235, 0.5531354017391854, 0.2504405001085667, 0.4576723538527239, 0.02907589449572949, 0.02391283754439626, 0.6815508112541282, 0.4101819936912841, 0.1291320664165603, 0.624364866051024, 0.9983414238854471, 0.3645041768280023, 0.8201605375869307, 0.4666105446851559, 0.0548118887496996, 23.28169327304722, 0.380539987933107, 0.9332820939178124, 0.2812503813563353, 0.8620038684368766, 0.1462057208942748, 0.832663076360586, 0.2694312632080076, 0.4641222360447021, 0.708517565327658, 0.2084573144612344, 0.7962269804755528, 0.8325326254736197},
{0.8732170204822794, 0.9703522590694529, 0.4688713205023043, 0.104077390692899, 0.9893130873462439, 0.8782113148179301, 0.3465927380552877, 0.01046630271381903, 0.3540369134480302, 0.9800683373980577, 0.9465493787120213, 0.3798801647584766, 0.3658999531291825, 0.9097369120901089, 0.6310969532646443, 0.1418073229841496, 0.05958331098322638, 0.5805958840653274, 0.2738276293487468, 0.08513152394054339, 26.32788781487411, 0.06444095187496028, 0.4517630461976831, 0.3211657842593503, 0.4991490433820918, 0.2301560980577137, 0.4739301594785909, 0.3796066473708281, 0.2941063130178673, 0.6618183254684047, 0.2466527787521567, 0.8276039473625254},
{0.7334065504866176, 0.1209511646366182, 0.108902520989654, 0.147561969493134, 0.1741231989000949, 0.3945930564784073, 0.01579730411623874, 0.3543509219367642, 0.8486318615777818, 0.8421190891628832, 0.2994339339994525, 0.9341792546540071, 0.9809843951611883, 0.8916205783197161, 0.05405090604211164, 0.701699561031048, 0.9594946179825994, 0.9566048351525434, 0.9147777540548568, 0.5770828560614391, 0.9351817447387765, 29.44270384933086, 0.7467412875329432, 0.7117697536948873, 0.5961703381872867, 0.5477195148634451, 0.4961482679582114, 0.8796888311436142, 0.02058798647305731, 0.7637097210179187, 0.187272323818762, 0.3451787774821231},
{0.2231344968286129, 0.6826439438942561, 0.8072795485681091, 0.7597294197012614, 0.8218497662189755, 0.7747338256664865, 0.3322933497908093, 0.6735803887171637, 0.2963836356873487, 0.1367375681712511, 0.7464888919930237, 0.4043087460553345, 0.3923827426322052, 0.8159613619328315, 0.3075095173377401, 0.4480692758420787, 0.1729311128548635, 0.01049369584311881, 0.8483167043186906, 0.2108393277493643, 0.7677084213080634, 0.9160499829256111, 31.18418178583161, 0.2083803753659384, 0.6222395093873411, 0.3136787178654094, 0.7895516975041856, 0.8024071458301418, 0.1609783629721528, 0.4650083899787478, 0.6443945351473671, 0.4254202493981888},
{0.5274886737824364, 0.2545214001230884, 0.08803923710970833, 0.9568209031502477, 0.6910479666629009, 0.2846496619714263, 0.3898416521268032, 0.3262573020118892, 0.4711508256584289, 0.6774618473217001, 0.599918344639385, 0.2515376012620413, 0.8610224146632673, 0.9644106655853596, 0.1037677687416795, 0.4016690458737055, 0.1194993979504065, 0.3220811018248556, 0.09127847131831117, 0.5281891794038818, 0.08812806188921689, 0.4884266024518979, 0.05565607364993534, 38.15066403949394, 0.9960807465958844, 0.3915313778528945, 0.8695869724784497, 0.4404522649247649, 0.8302419697596208, 0.07257763456045722, 0.2755498988113218, 0.2754820929169413},
{0.5332162684969857, 0.2889945456303152, 0.3518721769713227, 0.04244500967768565, 0.5863995741577644, 0.505946846949029, 0.2926133150521542, 0.09861573140020567, 0.7430423134576787, 0.3294216093287421, 0.5135333110224046, 0.9476428486743221, 0.5713859146171189, 0.769591431201935, 0.888193164377472, 0.3385239838949652, 0.8810238647091825, 0.7835823994079751, 0.4227008241202215, 0.783833305113399, 0.4725783954869833, 0.4509599574519822, 0.7623568558696128, 0.4812322596306657, 26.73958804933762, 0.746465960619353, 0.3389298280968987, 0.3699220297211774, 0.8152666224046701, 0.550246033213324, 0.05676937467036773, 0.8556632180031104},
{0.5256882831785423, 0.1540173049614769, 0.5519592833127245, 0.2954958220262348, 0.6744189138014833, 0.3985201133847381, 0.5555819238253875, 0.4262559819290951, 0.01798404032281891, 0.703835334189413, 0.6310595316756068, 0.3226089339437443, 0.6364300366144706, 0.3140270805886063, 0.6530497341813238, 0.5424382023320454, 0.01482453498541774, 0.7924641730623956, 0.6073481258617504, 0.4899820761460076, 0.6654545403707453, 0.02950367875734365, 0.7046417530685476, 0.6355539036056916, 0.6001295106219697, 32.82606511056387, 0.02119867459637685, 0.6675268276757012, 0.07850755893580934, 0.8527418385365269, 0.6399384035950921, 0.2656517744481785},
{0.211482720803167, 0.3531985085613678, 0.8198844220804503, 0.9329855954230764, 0.6346795519420854, 0.4834976396847546, 0.440997218176011, 0.838373472005185, 0.1085570889533741, 0.7702684533536599, 0.4958521254835985, 0.3014703462031903, 0.5061702022178564, 0.9539168726286053, 0.2060267824730314, 0.301283283382678, 0.6253110764719559, 0.1071916014870784, 0.62604172649629, 0.1036270528251653, 0.6038509282299837, 0.3879197826547685, 0.6095199947687829, 0.6125495511336818, 0.6339251379927912, 0.5797474879759662, 20.518555414391, 0.3071789812640437, 0.5587474156184588, 0.8599355339490155, 0.9441567789401004, 0.6921445919012273},
{0.6274222220249509, 0.1495092641156091, 0.363963672610411, 0.09930246277960557, 0.7279022639581763, 0.08947776034477033, 0.2162309795650294, 0.9202051768450284, 0.7587303021275993, 0.8082934737052796, 0.3126956714140547, 0.4124541067531398, 0.2223762669210046, 0.5668713265977853, 0.3448436322471007, 0.2795365815137779, 0.8274401978621143, 0.8786920808267425, 0.1943748381040053, 0.1413002099694472, 0.7142584212937612, 0.4535670590988038, 0.8486046066460203, 0.4062981395049745, 0.3516379311855637, 0.6223615258609673, 0.4433302783431803, 38.136833673079, 0.3443029687480897, 0.04146085909302925, 0.07937125266193894, 0.4152053032154745},
{0.2040989665939916, 0.02264812259025644, 0.8306581827882195, 0.5711473245914044, 0.9674114546632784, 0.1525254433460569, 0.2711760973479473, 0.2168531620059538, 0.06259017648462122, 0.7526152073046312, 0.5439457066828282, 0.5141036201887009, 0.9265636102820788, 0.2129325498896074, 0.08511613043268111, 0.6676964960723593, 0.7111060174497558, 0.6264913316144709, 0.118710315193798, 0.8263578912333505, 0.535903576341453, 0.4207868813084922, 0.8541577559048339, 0.3476169552576301, 0.6407335734602961, 0.07527448527694162, 0.1029608905226091, 0.6674289066552381, 23.77612353068125, 0.8222574292739269, 0.4177847298937293, 0.1120243122103961},
{0.9214015346766015, 0.5256495677016831, 0.9041926331600302, 0.479088533933307, 0.36170762944393, 0.418840296292349, 0.3713094211861552, 0.894010105557954, 0.884158907179213, 0.706388933238861, 0.9864357471592473, 0.5229565546554025, 0.898180822298996, 0.5297763786658372, 0.6162519300032682, 0.3602358117300149, 0.8473524429508452, 0.220663798530021, 0.8954286966979897, 0.734479565658848, 0.07350310473415582, 0.7397807309951063, 0.2986041142820746, 0.5602236219285399, 0.008669696894563473, 0.1485509443928323, 0.6170171114879124, 0.8922078660174843, 0.08604337900390029, 30.75076595955216, 0.9837892871651108, 0.8780584932888679},
{0.6039943579773142, 0.491023661037091, 0.6930962788010213, 0.4509316074013086, 0.244600792194598, 0.8652157622502296, 0.7601720652055538, 0.06824795015991762, 0.8614650425535108, 0.7973476662036476, 0.5274706153650589, 0.06699832486680694, 0.9830608826557091, 0.02411954845448376, 0.9159948096117938, 0.05989731169111805, 0.9964875625056208, 0.5490973316455776, 0.3070086180565647, 0.7847318988549384, 0.8389293638646981, 0.2995305785402646, 0.3901288004147664, 0.4046729912287069, 0.7000476578491556, 0.4910617396831564, 0.1912313022512377, 0.3246861286602927, 0.8834795174729793, 0.6414066459790291, 27.48500407990493, 0.1349742035494014},
{0.7582694728504996, 0.4238600361298778, 0.5519527173914217, 0.3998436860479148, 0.4997443859326505, 0.3650075322293619, 0.1414681892584472, 0.8737298720377492, 0.2430767928433392, 0.3774901321195912, 0.03245202587119216, 0.9489664031942351, 0.6380737545785872, 0.4558598670266945, 0.6433490584605778, 0.3348668942268838, 0.02219320526361806, 0.3020651497765501, 0.9591040517241186, 0.1465647044401787, 0.385614089905751, 0.904470358474136, 0.9768985463685137, 0.4305267716599943, 0.565639439236404, 0.9082976282050029, 0.6624017552226056, 0.3799009267903909, 0.8271982316429014, 0.8291980588143636, 0.7540017772884831, 37.43434063040866}};


const double b[M] = {
 43.51938327210452,
 19.86157688458145,
 32.80827104798441,
 21.52992167449894,
 37.68749680505844,
 28.76731740430303,
 18.57339306101101,
 12.39468180122769,
 9.904229907656406,
 16.92878521085951,
 27.57413026881049,
 9.947704776785255,
 34.79478795766441,
 29.14771957206293,
 25.41898295485367,
 37.84344829927709,
 28.53393771812673,
 11.10128169273699,
 19.83134032368699,
 8.670211364114987,
 27.62355076108517,
 25.40637735714781,
 33.97505169470374,
 14.86309799727051,
 29.49843653500105,
 16.90865895550604,
 24.61097519385071,
 8.724087823644748,
 29.32731083523764,
 18.05019545689942,
 10.65199284799313,
 27.93037740459069
};

const double expected_x[M] = {
 0.9834410263130171,
 0.4588578177311285,
 0.8754866467186926,
 0.3352646871053129,
 0.913972997414356,
 0.7463676433387144,
 0.3026743341026017,
 0.1310604182446916,
 0.05843357150535582,
 0.3726279803841743,
 0.6845393333064778,
 0.05150788208448213,
 0.7454040101388821,
 0.7819066012926237,
 0.4349881504878882,
 0.927876591558535,
 0.653333978025372,
 0.08626529845154844,
 0.3708912829526834,
 0.02395576831072903,
 0.734246156343295,
 0.5861947952299488,
 0.8180761332422043,
 0.1887838787271688,
 0.7925521678901373,
 0.2992883577871879,
 0.7822050139268599,
 0.04071519045442477,
 0.9221507942319959,
 0.3068234548252524,
 0.07583868976174361,
 0.5186699880232961
};

#endif
