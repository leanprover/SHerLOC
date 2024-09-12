"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x5xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x5xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<4x5xcomplex<f32>>
    %4 = "stablehlo.reverse"(%2) <{dimensions = array<i64: 0>}> : (tensor<4x5xcomplex<f32>>) -> tensor<4x5xcomplex<f32>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x5xcomplex<f32>>, tensor<4x5xcomplex<f32>>) -> ()
    "func.return"(%4) : (tensor<4x5xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(1.64156401,2.43617129), (-0.821430087,2.41240335), (6.91338158,2.4461894), (1.46334195,1.03390598), (3.88088608,-0.164413869)], [(-0.445322603,5.50484324), (-5.34640217,-4.61504793), (-2.96519136,-0.0412336588), (3.10464835,1.50602651), (0.975138604,-3.05653954)], [(-1.34973919,-1.22588968), (-0.13968581,-0.337649882), (4.62552118,2.57984948), (-0.534202218,2.66017938), (-0.312704831,1.53405464)], [(-0.866462588,0.111736268), (-2.33610892,-5.868590e-01), (5.45916462,4.405040e+00), (6.02274752,3.01540041), (-0.704937577,-3.68470764)]]> : tensor<4x5xcomplex<f32>>}> : () -> tensor<4x5xcomplex<f32>>
    "func.return"(%1) : (tensor<4x5xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-0.866462588,0.111736268), (-2.33610892,-5.868590e-01), (5.45916462,4.405040e+00), (6.02274752,3.01540041), (-0.704937577,-3.68470764)], [(-1.34973919,-1.22588968), (-0.13968581,-0.337649882), (4.62552118,2.57984948), (-0.534202218,2.66017938), (-0.312704831,1.53405464)], [(-0.445322603,5.50484324), (-5.34640217,-4.61504793), (-2.96519136,-0.0412336588), (3.10464835,1.50602651), (0.975138604,-3.05653954)], [(1.64156401,2.43617129), (-0.821430087,2.41240335), (6.91338158,2.4461894), (1.46334195,1.03390598), (3.88088608,-0.164413869)]]> : tensor<4x5xcomplex<f32>>}> : () -> tensor<4x5xcomplex<f32>>
    "func.return"(%0) : (tensor<4x5xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

