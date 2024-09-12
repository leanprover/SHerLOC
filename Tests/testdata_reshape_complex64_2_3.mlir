"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xcomplex<f32>>
    %4 = "stablehlo.reshape"(%2) : (tensor<2x3xcomplex<f32>>) -> tensor<3x2xcomplex<f32>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> ()
    "func.return"(%4) : (tensor<3x2xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(0.617260575,-0.0616655722), (-5.14312601,0.740792274), (-8.860930e+00,-2.22399831)], [(0.265819639,2.97546291), (1.72794175,7.77228832), (0.378611773,0.622524082)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%1) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(0.617260575,-0.0616655722), (-5.14312601,0.740792274)], [(-8.860930e+00,-2.22399831), (0.265819639,2.97546291)], [(1.72794175,7.77228832), (0.378611773,0.622524082)]]> : tensor<3x2xcomplex<f32>>}> : () -> tensor<3x2xcomplex<f32>>
    "func.return"(%0) : (tensor<3x2xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

