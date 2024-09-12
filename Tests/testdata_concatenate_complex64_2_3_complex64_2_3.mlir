"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x3xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x3xcomplex<f32>>
    %5 = "stablehlo.concatenate"(%3#0, %3#1) <{dimension = 0 : i64}> : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<4x3xcomplex<f32>>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x3xcomplex<f32>>, tensor<4x3xcomplex<f32>>) -> ()
    "func.return"(%5) : (tensor<4x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(1.07382131,-2.67878866), (0.427812099,4.57176638), (5.087150e+00,-0.220292255)], [(1.57771051,2.09996724), (2.37849593,-3.05391335), (0.732661545,-5.76225805)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    %2 = "stablehlo.constant"() <{value = dense<[[(-1.83474767,2.32461143), (-0.989446878,-0.815832734), (5.39325857,2.16857696)], [(2.27715683,-0.268771797), (0.337234288,3.84130883), (-7.95383501,-0.700022578)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%1, %2) : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(1.07382131,-2.67878866), (0.427812099,4.57176638), (5.087150e+00,-0.220292255)], [(1.57771051,2.09996724), (2.37849593,-3.05391335), (0.732661545,-5.76225805)], [(-1.83474767,2.32461143), (-0.989446878,-0.815832734), (5.39325857,2.16857696)], [(2.27715683,-0.268771797), (0.337234288,3.84130883), (-7.95383501,-0.700022578)]]> : tensor<4x3xcomplex<f32>>}> : () -> tensor<4x3xcomplex<f32>>
    "func.return"(%0) : (tensor<4x3xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

