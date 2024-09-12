"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x2xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %94 = "func.call"() <{callee = @expected}> : () -> tensor<2x2xui32>
    %95 = "func.call"() <{callee = @wrap_and_split}> : () -> tensor<2x2xui32>
    "stablehlo.custom_call"(%95, %94) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x2xui32>, tensor<2x2xui32>) -> ()
    "func.return"(%95) : (tensor<2x2xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x2xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %93 = "stablehlo.constant"() <{value = dense<[[2465931498, 3679230171], [255383827, 267815257]]> : tensor<2x2xui32>}> : () -> tensor<2x2xui32>
    "func.return"(%93) : (tensor<2x2xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x2xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "wrap_and_split", sym_visibility = "private"}> ({
    %81 = "stablehlo.constant"() <{value = dense<42> : tensor<i64>}> : () -> tensor<i64>
    %82 = "stablehlo.constant"() <{value = dense<32> : tensor<i64>}> : () -> tensor<i64>
    %83 = "stablehlo.shift_right_logical"(%81, %82) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %84 = "stablehlo.convert"(%83) : (tensor<i64>) -> tensor<ui32>
    %85 = "stablehlo.broadcast_in_dim"(%84) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %86 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %87 = "stablehlo.convert"(%86) : (tensor<ui32>) -> tensor<i64>
    %88 = "stablehlo.and"(%81, %87) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %89 = "stablehlo.convert"(%88) : (tensor<i64>) -> tensor<ui32>
    %90 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %91 = "stablehlo.concatenate"(%85, %90) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %92 = "func.call"(%91) <{callee = @_threefry_split}> : (tensor<2xui32>) -> tensor<2x2xui32>
    "func.return"(%92) : (tensor<2x2xui32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<2x2xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_threefry_split", sym_visibility = "private"}> ({
  ^bb0(%arg8: tensor<2xui32>):
    %55 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xui32>
    %56 = "stablehlo.slice"(%arg8) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %57 = "stablehlo.reshape"(%56) : (tensor<1xui32>) -> tensor<ui32>
    %58 = "stablehlo.slice"(%arg8) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %59 = "stablehlo.reshape"(%58) : (tensor<1xui32>) -> tensor<ui32>
    %60 = "stablehlo.slice"(%55) <{limit_indices = array<i64: 2>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<2xui32>
    %61 = "stablehlo.slice"(%55) <{limit_indices = array<i64: 4>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<2xui32>
    %62 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %63 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %64 = "stablehlo.xor"(%57, %59) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %65 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %66 = "stablehlo.xor"(%64, %65) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %67 = "stablehlo.broadcast_in_dim"(%57) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %68 = "stablehlo.add"(%60, %67) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %69 = "stablehlo.broadcast_in_dim"(%59) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %70 = "stablehlo.add"(%61, %69) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %71 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %72 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %73:9 = "stablehlo.while"(%72, %71, %68, %70, %59, %66, %57, %62, %63) ({
    ^bb0(%arg18: tensor<i64>, %arg19: tensor<i64>, %arg20: tensor<2xui32>, %arg21: tensor<2xui32>, %arg22: tensor<ui32>, %arg23: tensor<ui32>, %arg24: tensor<ui32>, %arg25: tensor<4xui32>, %arg26: tensor<4xui32>):
      %79 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %80 = "stablehlo.compare"(%arg18, %79) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%80) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg9: tensor<i64>, %arg10: tensor<i64>, %arg11: tensor<2xui32>, %arg12: tensor<2xui32>, %arg13: tensor<ui32>, %arg14: tensor<ui32>, %arg15: tensor<ui32>, %arg16: tensor<4xui32>, %arg17: tensor<4xui32>):
      %76:8 = "func.call"(%arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17) <{callee = @None}> : (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %77 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %78 = "stablehlo.add"(%arg9, %77) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%78, %76#0, %76#1, %76#2, %76#3, %76#4, %76#5, %76#6, %76#7) : (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %74 = "stablehlo.concatenate"(%73#2, %73#3) <{dimension = 0 : i64}> : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %75 = "stablehlo.reshape"(%74) : (tensor<4xui32>) -> tensor<2x2xui32>
    "func.return"(%75) : (tensor<2x2xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<2xui32>, %arg2: tensor<2xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.add"(%arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %3 = "stablehlo.reshape"(%2) : (tensor<1xui32>) -> tensor<ui32>
    %4 = "stablehlo.add"(%arg1, %arg2) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %5 = "stablehlo.broadcast_in_dim"(%3) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %6 = "stablehlo.shift_left"(%arg2, %5) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %7 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %8 = "stablehlo.subtract"(%7, %3) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %9 = "stablehlo.broadcast_in_dim"(%8) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %10 = "stablehlo.shift_right_logical"(%arg2, %9) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %11 = "stablehlo.or"(%6, %10) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %12 = "stablehlo.xor"(%4, %11) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %13 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %14 = "stablehlo.reshape"(%13) : (tensor<1xui32>) -> tensor<ui32>
    %15 = "stablehlo.add"(%4, %12) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %16 = "stablehlo.broadcast_in_dim"(%14) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %17 = "stablehlo.shift_left"(%12, %16) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %18 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %19 = "stablehlo.subtract"(%18, %14) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %20 = "stablehlo.broadcast_in_dim"(%19) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %21 = "stablehlo.shift_right_logical"(%12, %20) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %22 = "stablehlo.or"(%17, %21) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %23 = "stablehlo.xor"(%15, %22) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %24 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %25 = "stablehlo.reshape"(%24) : (tensor<1xui32>) -> tensor<ui32>
    %26 = "stablehlo.add"(%15, %23) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %27 = "stablehlo.broadcast_in_dim"(%25) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %28 = "stablehlo.shift_left"(%23, %27) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %29 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %30 = "stablehlo.subtract"(%29, %25) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %31 = "stablehlo.broadcast_in_dim"(%30) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %32 = "stablehlo.shift_right_logical"(%23, %31) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %33 = "stablehlo.or"(%28, %32) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %34 = "stablehlo.xor"(%26, %33) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %35 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %36 = "stablehlo.reshape"(%35) : (tensor<1xui32>) -> tensor<ui32>
    %37 = "stablehlo.add"(%26, %34) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %38 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %39 = "stablehlo.shift_left"(%34, %38) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %40 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %41 = "stablehlo.subtract"(%40, %36) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %42 = "stablehlo.broadcast_in_dim"(%41) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %43 = "stablehlo.shift_right_logical"(%34, %42) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %44 = "stablehlo.or"(%39, %43) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %45 = "stablehlo.xor"(%37, %44) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %46 = "stablehlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %47 = "stablehlo.add"(%37, %46) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %48 = "stablehlo.broadcast_in_dim"(%arg4) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %49 = "stablehlo.add"(%45, %48) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %50 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %51 = "stablehlo.add"(%arg0, %50) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %52 = "stablehlo.convert"(%51) : (tensor<i64>) -> tensor<ui32>
    %53 = "stablehlo.broadcast_in_dim"(%52) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %54 = "stablehlo.add"(%49, %53) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    "func.return"(%1, %47, %54, %arg4, %arg5, %arg3, %arg7, %arg6) : (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

