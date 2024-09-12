"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi16>, tensor<3x6xcomplex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f32>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi16>) -> tensor<4x3xcomplex<f32>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f32>>) -> tensor<3x6xcomplex<f32>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi16>, tensor<3x6xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1, 0, 0], [-3, -2, 3], [0, 0, 1], [2, -3, 0]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[(1.96556449,3.60451722), (0.431844205,-1.44997811), (3.5025692,3.78915787), (-1.14176381,0.982468426), (0.568919897,-2.3634913), (-4.42918444,-1.0014993)], [(6.6052928,0.712479591), (-2.61274672,-0.759193062), (-3.73786402,-3.17064571), (-3.7231698,-1.29486406), (-6.35924864,-1.75780439), (0.864134073,-3.63506389)], [(1.6930505,0.0291357301), (3.46606898,0.467725307), (-1.72243237,-5.48474503), (-3.47399879,1.28193557), (-2.03778362,-3.380370e+00), (2.06004691,-3.98042583)]]> : tensor<3x6xcomplex<f32>>}> : () -> tensor<3x6xcomplex<f32>>
    "func.return"(%1, %2) : (tensor<4x3xi16>, tensor<3x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-1.96556449,-3.60451722), (-0.431844205,1.44997811), (-3.5025692,-3.78915787), (1.14176381,-0.982468426), (-0.568919897,2.3634913), (4.42918444,1.0014993)], [(-14.0281277,-12.151104), (14.328167,7.2714963), (-8.19927692,-21.4804173), (0.449635506,3.48812938), (4.89838696,0.46497345), (17.7394257,-1.66665173)], [(1.6930505,0.0291357301), (3.46606898,0.467725307), (-1.72243237,-5.48474503), (-3.47399879,1.28193557), (-2.03778362,-3.380370e+00), (2.06004691,-3.98042583)], [(-15.8847485,5.07159567), (8.70192813,-0.622376918), (18.2187309,17.0902519), (8.88598251,5.84952879), (20.2155857,0.546430588), (-11.4507713,8.90219306)]]> : tensor<4x6xcomplex<f32>>}> : () -> tensor<4x6xcomplex<f32>>
    "func.return"(%0) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

