"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xcomplex<f32>>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xcomplex<f32>>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %5 = "stablehlo.real"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
      %6 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %7 = "stablehlo.compare"(%5, %6) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %9 = "stablehlo.select"(%7, %8, %5) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %10 = "stablehlo.compare"(%5, %5) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %11 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
      %12 = "stablehlo.select"(%10, %11, %9) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %13 = "stablehlo.imag"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
      %14 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %15 = "stablehlo.compare"(%13, %14) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %16 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %17 = "stablehlo.select"(%15, %16, %13) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %18 = "stablehlo.compare"(%13, %13) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %19 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
      %20 = "stablehlo.select"(%18, %19, %17) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %21 = "stablehlo.real"(%arg1) : (tensor<complex<f32>>) -> tensor<f32>
      %22 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %23 = "stablehlo.compare"(%21, %22) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %24 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %25 = "stablehlo.select"(%23, %24, %21) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %26 = "stablehlo.compare"(%21, %21) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %27 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
      %28 = "stablehlo.select"(%26, %27, %25) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %29 = "stablehlo.imag"(%arg1) : (tensor<complex<f32>>) -> tensor<f32>
      %30 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %31 = "stablehlo.compare"(%29, %30) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %32 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %33 = "stablehlo.select"(%31, %32, %29) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %34 = "stablehlo.compare"(%29, %29) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %35 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
      %36 = "stablehlo.select"(%34, %35, %33) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %37 = "stablehlo.compare"(%20, %36) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %38 = "stablehlo.compare"(%12, %28) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %39 = "stablehlo.compare"(%12, %28) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %40 = "stablehlo.and"(%39, %37) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %41 = "stablehlo.or"(%38, %40) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%41) : (tensor<i1>) -> ()
    }) : (tensor<5x7xcomplex<f32>>) -> tensor<5x7xcomplex<f32>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xcomplex<f32>>, tensor<5x7xcomplex<f32>>) -> ()
    "func.return"(%4) : (tensor<5x7xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-4.230590e+00,-0.931078791), (2.437567,6.385600e+00), (-3.22736883,1.83972192), (-2.1802094,0.889342188), (2.12194395,-0.191629708), (5.18555069,-4.99879742), (3.67927694,-2.2557056)], [(0.379014701,0.71961534), (3.83116126,-0.202661604), (-1.31466472,-0.621450722), (-0.886600852,-3.84756875), (-0.77832818,-2.22487688), (-0.806218206,-2.79777122), (-1.17338157,-0.0607901253)], [(-1.40947139,0.967940688), (3.73917222,1.84880078), (0.479227781,3.7598331), (-0.0535792634,-1.28269935), (-1.52044451,0.930376708), (-4.65916061,3.13317561), (-1.79412854,-2.32595181)], [(0.751593947,-2.26882839), (-1.97411048,-5.05067635), (-2.80797291,0.160113797), (0.709535956,-0.731431484), (-0.599625289,1.46191716), (-3.644970e+00,-0.755360901), (-0.0846063494,-2.87114239)], [(3.9562633,-2.32443857), (2.38367414,1.16753387), (-1.4873457,-1.95769823), (-1.39860535,2.15638304), (-1.28114533,-3.99063683), (4.16027069,-5.125950e+00), (-2.47642064,0.641588687)]]> : tensor<5x7xcomplex<f32>>}> : () -> tensor<5x7xcomplex<f32>>
    "func.return"(%1) : (tensor<5x7xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-4.230590e+00,-0.931078791), (-1.97411048,-5.05067635), (-3.22736883,1.83972192), (-2.1802094,0.889342188), (-1.52044451,0.930376708), (-4.65916061,3.13317561), (-2.47642064,0.641588687)], [(-1.40947139,0.967940688), (2.38367414,1.16753387), (-2.80797291,0.160113797), (-1.39860535,2.15638304), (-1.28114533,-3.99063683), (-3.644970e+00,-0.755360901), (-1.79412854,-2.32595181)], [(0.379014701,0.71961534), (2.437567,6.385600e+00), (-1.4873457,-1.95769823), (-0.886600852,-3.84756875), (-0.77832818,-2.22487688), (-0.806218206,-2.79777122), (-1.17338157,-0.0607901253)], [(0.751593947,-2.26882839), (3.73917222,1.84880078), (-1.31466472,-0.621450722), (-0.0535792634,-1.28269935), (-0.599625289,1.46191716), (4.16027069,-5.125950e+00), (-0.0846063494,-2.87114239)], [(3.9562633,-2.32443857), (3.83116126,-0.202661604), (0.479227781,3.7598331), (0.709535956,-0.731431484), (2.12194395,-0.191629708), (5.18555069,-4.99879742), (3.67927694,-2.2557056)]]> : tensor<5x7xcomplex<f32>>}> : () -> tensor<5x7xcomplex<f32>>
    "func.return"(%0) : (tensor<5x7xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

