"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f64>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xcomplex<f32>>) -> tensor<4x3xcomplex<f64>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f64>>) -> tensor<3x6xcomplex<f64>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f64>>, tensor<3x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<4x6xcomplex<f64>>, tensor<4x6xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-0.3539837,0.00281779235), (-0.785270512,-4.94724798), (-2.93613601,3.02366447)], [(2.44117117,4.66270542), (1.46640646,1.00852525), (-3.38045692,-2.52066016)], [(2.29947853,0.0652045608), (3.18498302,0.968257784), (-2.87217474,-4.0397296)], [(-2.86522126,-0.381847203), (-2.04735804,-7.29907799), (3.32911158,-2.25939536)]]> : tensor<4x3xcomplex<f32>>}> : () -> tensor<4x3xcomplex<f32>>
    %2 = "stablehlo.constant"() <{value = dense<[[(0.019432959080136585,0.027617238370666406), (1.7830913820737775,-3.0560207968723949), (5.7261039743173985,6.3825088711446085), (-0.34909617477890459,1.9563861015464501), (-3.6534320469875681,-3.989383334024033), (-5.514935312153348,-2.9894504081124946)], [(-1.3766765028504002,4.8164311464060274), (-2.2256501958163013,-4.0911920171211609), (1.6014621979172141,2.6768734906309857), (-2.0295330613427929,-8.6448614744724921), (-1.12750177534615,-0.99057826801226778), (-0.34322032963882654,5.3078260673315825)], [(-4.4754531542325511,0.63788285147780388), (-3.0335743039140661,3.8780017610533406), (-0.083369366570918105,-3.9831179306777731), (-0.33778772111732469,-0.30794998351893749), (-2.0341102097710291,2.5418700897413031), (-3.0350747348330556,-1.814868119104732)]]> : tensor<3x6xcomplex<f64>>}> : () -> tensor<3x6xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(36.113981399878206,-12.386342114764373), (-21.933767447399653,-5.2485096390966746), (22.229039860490037,-0.82517459336226118), (-39.133551236268495,16.018471970184251), (-4.4240838517600842,-5.8559582267422305), (42.888217935073385,-5.2757544562575482)], [(9.7793708685858185,14.957222291978438), (39.494510446771599,-12.853022864887294), (-25.890918910126395,61.495336778186953), (-3.8661753150163056,-9.6831156680442803), (22.311726239086997,-32.828731154047787), (0.30492204297032188,-11.789519365892966)], [(6.4258044031065662,30.319555563716055), (25.551151265558669,-20.979847540735427), (-0.59165722656135245,36.903235462011267), (0.70227233489375251,-22.773885756416178), (5.3379863386579816,-12.741848939468031), (-17.333355844553477,26.812733745760376)], [(24.470868750181019,12.336361505667099), (-32.918279257960393,52.46098077761603), (-6.9864539234565779,-50.715392489026833), (-59.017372706149779,26.778665201663195), (2.9939577451263055,36.141338837600827), (39.90032090606266,3.125006405164628)]]> : tensor<4x6xcomplex<f64>>}> : () -> tensor<4x6xcomplex<f64>>
    "func.return"(%0) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

