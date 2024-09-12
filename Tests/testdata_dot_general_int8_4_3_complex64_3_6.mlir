"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi8>, tensor<3x6xcomplex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f32>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi8>) -> tensor<4x3xcomplex<f32>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f32>>) -> tensor<3x6xcomplex<f32>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi8>, tensor<3x6xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[4, 0, -1], [2, 1, 0], [0, 1, 0], [-3, -4, 1]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[(3.40995479,-0.0257220604), (-2.53519201,2.03951287), (-4.97211123,1.81478179), (-3.52636361,-1.93579137), (5.29525232,4.554245), (1.00284398,-5.50234175)], [(1.02853799,0.140062451), (2.36836863,5.45560789), (-2.04044294,-0.172459513), (-0.694677591,-0.294079036), (3.63483596,-1.58528793), (1.45537043,-5.07602644)], [(-0.801572144,2.54920435), (-5.65842485,-0.998069345), (1.37239444,-4.18741512), (-1.0028218,2.10472584), (-2.48979735,-1.17930412), (-0.467441708,-3.73676658)]]> : tensor<3x6xcomplex<f32>>}> : () -> tensor<3x6xcomplex<f32>>
    "func.return"(%1, %2) : (tensor<4x3xi8>, tensor<3x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(14.441391,-2.6520927), (-4.4823432,9.15612125), (-21.2608395,11.4465427), (-13.1026325,-9.84789085), (23.6708069,19.3962841), (4.47881746,-1.827260e+01)], [(7.84844779,0.0886183306), (-2.7020154,9.53463363), (-11.9846649,3.45710397), (-7.74740505,-4.16566181), (14.2253408,7.52320194), (3.46105838,-16.0807095)], [(1.02853799,0.140062451), (2.36836863,5.45560789), (-2.04044294,-0.172459513), (-0.694677591,-0.294079036), (3.63483596,-1.58528793), (1.45537043,-5.07602644)], [(-15.1455879,2.06612062), (-7.52632331,-28.9390411), (2.445050e+01,-8.94192314), (12.3549795,9.08841609), (-32.9148979,-8.50088691), (-9.29745483,33.0743637)]]> : tensor<4x6xcomplex<f32>>}> : () -> tensor<4x6xcomplex<f32>>
    "func.return"(%0) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

