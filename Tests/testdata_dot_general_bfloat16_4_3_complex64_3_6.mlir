"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xbf16>, tensor<3x6xcomplex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f32>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xbf16>) -> tensor<4x3xcomplex<f32>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f32>>) -> tensor<3x6xcomplex<f32>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xbf16>, tensor<3x6xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.203130e+00, 3.281250e+00, -2.062500e+00], [-1.648440e+00, 2.218750e+00, 1.304690e+00], [-1.140630e+00, 2.406250e+00, 2.636720e-01], [-1.210940e+00, 9.257810e-01, -1.671880e+00]]> : tensor<4x3xbf16>}> : () -> tensor<4x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[(-1.23465157,-1.97534931), (1.670120e+00,0.216153592), (-5.166550e+00,-0.334587127), (0.912404239,0.554522038), (4.53575134,-0.999703705), (-0.30895409,-1.16958821)], [(-0.4028036,-2.49677873), (-3.14465165,-2.28515124), (0.610261738,7.095100e+00), (-2.31424093,1.29751551), (0.101109952,2.62985325), (-1.06779826,5.842440e+00)], [(1.69907737,3.1651814), (0.615705729,-2.47314763), (-0.0892849043,0.24179633), (-0.259796768,1.7544111), (-4.16017103,-3.20194054), (0.222340554,-2.38368416)]]> : tensor<3x6xcomplex<f32>>}> : () -> tensor<3x6xcomplex<f32>>
    "func.return"(%1, %2) : (tensor<4x3xbf16>, tensor<3x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-3.34060621,-12.3441505), (-13.5976439,-2.65734577), (8.4025774,23.1846409), (-8.15550899,-0.0281591415), (3.45504379,16.435976), (-3.59058022,25.4940147)], [(3.3582902,1.84608459), (-8.926980e+00,-8.653180e+00), (9.75426483,16.6092663), (-6.97771692,4.25372601), (-12.6802883,3.30540419), (-1.56980085,11.7809448)], [(0.887027144,-2.92017174), (-9.30945491,-6.397295), (7.33799696,17.5179768), (-6.67785406,2.95223379), (-6.02721596,6.62410975), (-2.15836382,14.7639227)], [(-1.71846724,-5.21123409), (-5.9630537,1.75749493), (6.97061157,6.56942177), (-2.81299782,-2.40343189), (1.55638027,8.99849128), (-0.986149132,10.8103418)]]> : tensor<4x6xcomplex<f32>>}> : () -> tensor<4x6xcomplex<f32>>
    "func.return"(%0) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

