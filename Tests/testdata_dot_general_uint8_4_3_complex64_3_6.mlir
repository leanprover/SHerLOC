"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui8>, tensor<3x6xcomplex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f32>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui8>) -> tensor<4x3xcomplex<f32>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f32>>) -> tensor<3x6xcomplex<f32>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui8>, tensor<3x6xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[3, 2, 0], [1, 4, 3], [1, 1, 5], [1, 3, 0]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[(-0.564038396,-0.775520741), (1.42231762,5.04936075), (1.62641478,-6.75465869), (4.85501909,0.00674611516), (4.32904959,-1.54604387), (3.03411436,0.751372754)], [(2.87488794,-0.15537025), (6.07330322,0.569996715), (4.11665964,-2.51806188), (1.43245494,-0.167148665), (1.91975379,2.72274685), (4.06666088,-2.98041606)], [(0.0976789072,-3.41094756), (2.66272306,3.53814578), (0.938251554,-0.870848238), (-1.1626761,-1.09821248), (-1.71060216,4.5977397), (-0.723560869,-3.03641939)]]> : tensor<3x6xcomplex<f32>>}> : () -> tensor<3x6xcomplex<f32>>
    "func.return"(%1, %2) : (tensor<4x3xui8>, tensor<3x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(4.05766058,-2.63730264), (16.413559,16.2880764), (13.1125641,-2.530010e+01), (17.4299679,-3.140590e-01), (16.8266563,0.807362079), (17.2356644,-3.70671391)], [(11.22855,-11.6298447), (3.370370e+01,17.9437847), (20.9078083,-19.4394512), (7.09681034,-3.95648599), (6.87625789,23.1381626), (17.1300755,-20.2795506)], [(2.79924417,-17.9856281), (20.8092365,23.3100872), (10.4343319,-13.6269617), (0.474093914,-5.65146494), (-2.3042078,24.1654015), (3.48297095,-17.4111404)], [(8.06062602,-1.24163151), (19.6422272,6.75935077), (13.9763947,-14.3088446), (9.1523838,-0.494699895), (10.0883102,6.62219667), (15.2340965,-8.1898756)]]> : tensor<4x6xcomplex<f32>>}> : () -> tensor<4x6xcomplex<f32>>
    "func.return"(%0) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

