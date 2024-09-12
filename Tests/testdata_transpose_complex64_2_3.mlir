"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xcomplex<f32>>
    %4 = "stablehlo.transpose"(%2) <{permutation = array<i64: 1, 0>}> : (tensor<2x3xcomplex<f32>>) -> tensor<3x2xcomplex<f32>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> ()
    "func.return"(%4) : (tensor<3x2xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-5.08990383,2.25510144), (-5.3642478,-0.41310975), (-0.010698243,-0.417641759)], [(-5.15686893,-8.785930e-01), (2.19871163,-1.17818773), (-5.12186956,2.18995166)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%1) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-5.08990383,2.25510144), (-5.15686893,-8.785930e-01)], [(-5.3642478,-0.41310975), (2.19871163,-1.17818773)], [(-0.010698243,-0.417641759), (-5.12186956,2.18995166)]]> : tensor<3x2xcomplex<f32>>}> : () -> tensor<3x2xcomplex<f32>>
    "func.return"(%0) : (tensor<3x2xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

