"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x2xf32>, tensor<3x1xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xcomplex<f32>>
    %5 = "stablehlo.broadcast_in_dim"(%3#1) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<3x1xf32>) -> tensor<3x2xf32>
    %6 = "stablehlo.complex"(%3#0, %5) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xcomplex<f32>>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> ()
    "func.return"(%6) : (tensor<3x2xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x2xf32>, tensor<3x1xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[4.3779788, 0.256987363], [-1.94227421, -0.415562898], [-4.24635506, 0.812674164]]> : tensor<3x2xf32>}> : () -> tensor<3x2xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[-1.35233212], [-1.11680901], [-1.31206405]]> : tensor<3x1xf32>}> : () -> tensor<3x1xf32>
    "func.return"(%1, %2) : (tensor<3x2xf32>, tensor<3x1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(4.3779788,-1.35233212), (0.256987363,-1.35233212)], [(-1.94227421,-1.11680901), (-0.415562898,-1.11680901)], [(-4.24635506,-1.31206405), (0.812674164,-1.31206405)]]> : tensor<3x2xcomplex<f32>>}> : () -> tensor<3x2xcomplex<f32>>
    "func.return"(%0) : (tensor<3x2xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

