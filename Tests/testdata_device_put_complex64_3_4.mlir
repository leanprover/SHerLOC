"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xcomplex<f32>>
    "stablehlo.custom_call"(%2, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()
    "func.return"(%2) : (tensor<3x4xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-2.31154037,2.16501498), (-0.62014389,-3.54022598), (6.43273115,-0.0746307299), (-1.64691806,-2.16097689)], [(-4.05629301,-1.24208844), (1.9414624,0.0947651639), (-0.0559204295,3.14833903), (4.0608716,0.622570335)], [(0.526463509,8.46901607), (-0.513401508,-1.28572524), (-1.92279816,-2.3590281), (3.56461072,-6.23003673)]]> : tensor<3x4xcomplex<f32>>}> : () -> tensor<3x4xcomplex<f32>>
    "func.return"(%1) : (tensor<3x4xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-2.31154037,2.16501498), (-0.62014389,-3.54022598), (6.43273115,-0.0746307299), (-1.64691806,-2.16097689)], [(-4.05629301,-1.24208844), (1.9414624,0.0947651639), (-0.0559204295,3.14833903), (4.0608716,0.622570335)], [(0.526463509,8.46901607), (-0.513401508,-1.28572524), (-1.92279816,-2.3590281), (3.56461072,-6.23003673)]]> : tensor<3x4xcomplex<f32>>}> : () -> tensor<3x4xcomplex<f32>>
    "func.return"(%0) : (tensor<3x4xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

