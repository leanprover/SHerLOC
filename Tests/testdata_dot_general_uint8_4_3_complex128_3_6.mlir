"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui8>, tensor<3x6xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f64>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui8>) -> tensor<4x3xcomplex<f64>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f64>>) -> tensor<3x6xcomplex<f64>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f64>>, tensor<3x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f64>>, tensor<4x6xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui8>, tensor<3x6xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[3, 5, 1], [5, 4, 0], [2, 2, 0], [0, 5, 1]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[(-1.159777922081723,-2.1210600715975723), (-3.6759507480593769,2.8541588066099854), (-1.131871176693267,-0.42029801972190028), (-4.0814185938412404,1.6993778077858495), (4.6634238704536797,2.8118450010206213), (-5.5348687096391984,0.94758993278446801)], [(-1.447665848927524,-1.1311773261876905), (-1.5066190379356053,1.8164554293852022), (-7.0022451313377525,-0.67964635625214553), (0.44080956747846106,-1.4747596910496543), (2.4833349665368085,-1.8458026910797676), (-0.71772118634575954,1.757141688087382)], [(-2.268327543117409,2.3291689643496833), (-3.6273572969373804,-2.5444646367743919), (-0.085231008270121061,1.1802852995715614), (0.29442821387023965,3.7843548739802353), (-0.0035038380237289849,5.0421927028366635), (3.4785085672612288,-1.5842764900985573)]]> : tensor<3x6xcomplex<f64>>}> : () -> tensor<3x6xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<4x3xui8>, tensor<3x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-12.9859905540002,-9.6898978813814853), (-22.188304730793536,15.100288929981573), (-38.492070195038686,-3.478840540854867), (-9.7457797302611766,1.5086898420895123), (26.403442606021354,4.2487142504996882), (-16.714703493385162,10.044201748691755)], [(-11.589553006118711,-15.130009662738622), (-24.406229892039306,21.536615750590737), (-33.668336408817346,-4.8200755236180832), (-18.643854699292358,2.5978502747306296), (33.250459218415635,6.6760142407840366), (-30.545228293579029,11.766516416271868)], [(-5.2148875420184941,-6.5044747955705251), (-10.365139571989964,9.3412284719903749), (-16.26823261606204,-2.1998887519480919), (-7.281218052725559,0.44923623347239028), (14.293517673980976,1.9320846198817074), (-12.505179791969915,5.4094632417437003)], [(-9.5066567877550305,-3.326717666588769), (-11.160452486615407,6.5378125101516185), (-35.096456664958886,-2.2179464816891663), (2.498476051262545,-3.5894435812680365), (12.413170994660314,-4.1868207525621752), (-0.11009736446756868,7.2014319503383524)]]> : tensor<4x6xcomplex<f64>>}> : () -> tensor<4x6xcomplex<f64>>
    "func.return"(%0) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

