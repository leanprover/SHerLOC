"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<f64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %529 = "func.call"() <{callee = @inputs}> : () -> tensor<f32>
    %530 = "func.call"() <{callee = @expected}> : () -> tensor<f64>
    %531 = "func.call"(%529) <{callee = @"<lambda>"}> : (tensor<f32>) -> tensor<f64>
    "stablehlo.custom_call"(%531, %530) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<f64>, tensor<f64>) -> ()
    "func.return"(%531) : (tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<f32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %528 = "stablehlo.constant"() <{value = dense<-2.72118402> : tensor<f32>}> : () -> tensor<f32>
    "func.return"(%528) : (tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<f64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %527 = "stablehlo.constant"() <{value = dense<0xFFF8000000000000> : tensor<f64>}> : () -> tensor<f64>
    "func.return"(%527) : (tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<f32>) -> tensor<f64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "<lambda>", sym_visibility = "private"}> ({
  ^bb0(%arg165: tensor<f32>):
    %515 = "stablehlo.constant"() <{value = dense<42> : tensor<i64>}> : () -> tensor<i64>
    %516 = "stablehlo.constant"() <{value = dense<32> : tensor<i64>}> : () -> tensor<i64>
    %517 = "stablehlo.shift_right_logical"(%515, %516) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %518 = "stablehlo.convert"(%517) : (tensor<i64>) -> tensor<ui32>
    %519 = "stablehlo.broadcast_in_dim"(%518) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %520 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %521 = "stablehlo.convert"(%520) : (tensor<ui32>) -> tensor<i64>
    %522 = "stablehlo.and"(%515, %521) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %523 = "stablehlo.convert"(%522) : (tensor<i64>) -> tensor<ui32>
    %524 = "stablehlo.broadcast_in_dim"(%523) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %525 = "stablehlo.concatenate"(%519, %524) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %526 = "func.call"(%525, %arg165) <{callee = @_gamma}> : (tensor<2xui32>, tensor<f32>) -> tensor<f64>
    "func.return"(%526) : (tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<f32>) -> tensor<f64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_gamma", sym_visibility = "private"}> ({
  ^bb0(%arg155: tensor<2xui32>, %arg156: tensor<f32>):
    %475 = "stablehlo.convert"(%arg156) : (tensor<f32>) -> tensor<f64>
    %476 = "stablehlo.reshape"(%arg155) : (tensor<2xui32>) -> tensor<1x2xui32>
    %477 = "func.call"(%476) <{callee = @_threefry_split}> : (tensor<1x2xui32>) -> tensor<1x1x2xui32>
    %478 = "stablehlo.reshape"(%477) : (tensor<1x1x2xui32>) -> tensor<1x2xui32>
    %479 = "stablehlo.reshape"(%475) : (tensor<f64>) -> tensor<1xf64>
    %480 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %481 = "stablehlo.broadcast_in_dim"(%480) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %482 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %483:4 = "stablehlo.while"(%478, %479, %482, %481) ({
    ^bb0(%arg161: tensor<1x2xui32>, %arg162: tensor<1xf64>, %arg163: tensor<i64>, %arg164: tensor<1xf64>):
      %513 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %514 = "stablehlo.compare"(%arg163, %513) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%514) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg157: tensor<1x2xui32>, %arg158: tensor<1xf64>, %arg159: tensor<i64>, %arg160: tensor<1xf64>):
      %485 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %486 = "stablehlo.compare"(%arg159, %485) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %487 = "stablehlo.convert"(%arg159) : (tensor<i64>) -> tensor<i64>
      %488 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %489 = "stablehlo.add"(%487, %488) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %490 = "stablehlo.select"(%486, %489, %arg159) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
      %491 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %492 = "stablehlo.dynamic_slice"(%arg157, %490, %491) <{slice_sizes = array<i64: 1, 2>}> : (tensor<1x2xui32>, tensor<i64>, tensor<i64>) -> tensor<1x2xui32>
      %493 = "stablehlo.reshape"(%492) : (tensor<1x2xui32>) -> tensor<2xui32>
      %494 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %495 = "stablehlo.compare"(%arg159, %494) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %496 = "stablehlo.convert"(%arg159) : (tensor<i64>) -> tensor<i64>
      %497 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %498 = "stablehlo.add"(%496, %497) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %499 = "stablehlo.select"(%495, %498, %arg159) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
      %500 = "stablehlo.dynamic_slice"(%arg158, %499) <{slice_sizes = array<i64: 1>}> : (tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
      %501 = "stablehlo.reshape"(%500) : (tensor<1xf64>) -> tensor<f64>
      %502 = "func.call"(%493, %501) <{callee = @None_0}> : (tensor<2xui32>, tensor<f64>) -> tensor<f64>
      %503 = "stablehlo.broadcast_in_dim"(%502) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
      %504 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %505 = "stablehlo.compare"(%arg159, %504) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %506 = "stablehlo.convert"(%arg159) : (tensor<i64>) -> tensor<i64>
      %507 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %508 = "stablehlo.add"(%506, %507) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %509 = "stablehlo.select"(%505, %508, %arg159) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
      %510 = "stablehlo.dynamic_update_slice"(%arg160, %503, %509) : (tensor<1xf64>, tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
      %511 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %512 = "stablehlo.add"(%arg159, %511) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%arg157, %arg158, %512, %510) : (tensor<1x2xui32>, tensor<1xf64>, tensor<i64>, tensor<1xf64>) -> ()
    }) : (tensor<1x2xui32>, tensor<1xf64>, tensor<i64>, tensor<1xf64>) -> (tensor<1x2xui32>, tensor<1xf64>, tensor<i64>, tensor<1xf64>)
    %484 = "stablehlo.reshape"(%483#3) : (tensor<1xf64>) -> tensor<f64>
    "func.return"(%484) : (tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<1x2xui32>) -> tensor<1x1x2xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_threefry_split", sym_visibility = "private"}> ({
  ^bb0(%arg136: tensor<1x2xui32>):
    %446 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xui32>
    %447 = "stablehlo.slice"(%arg136) <{limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<1x2xui32>) -> tensor<1x1xui32>
    %448 = "stablehlo.reshape"(%447) : (tensor<1x1xui32>) -> tensor<1xui32>
    %449 = "stablehlo.slice"(%arg136) <{limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>}> : (tensor<1x2xui32>) -> tensor<1x1xui32>
    %450 = "stablehlo.reshape"(%449) : (tensor<1x1xui32>) -> tensor<1xui32>
    %451 = "stablehlo.slice"(%446) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %452 = "stablehlo.slice"(%446) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %453 = "stablehlo.broadcast_in_dim"(%451) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xui32>) -> tensor<1x1xui32>
    %454 = "stablehlo.broadcast_in_dim"(%452) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xui32>) -> tensor<1x1xui32>
    %455 = "stablehlo.broadcast_in_dim"(%448) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xui32>) -> tensor<1x1xui32>
    %456 = "stablehlo.broadcast_in_dim"(%450) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xui32>) -> tensor<1x1xui32>
    %457 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %458 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %459 = "stablehlo.xor"(%455, %456) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %460 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %461 = "stablehlo.broadcast_in_dim"(%460) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1x1xui32>
    %462 = "stablehlo.xor"(%459, %461) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %463 = "stablehlo.add"(%453, %455) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %464 = "stablehlo.add"(%454, %456) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %465 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %466 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %467:9 = "stablehlo.while"(%466, %465, %463, %464, %456, %462, %455, %457, %458) ({
    ^bb0(%arg146: tensor<i64>, %arg147: tensor<i64>, %arg148: tensor<1x1xui32>, %arg149: tensor<1x1xui32>, %arg150: tensor<1x1xui32>, %arg151: tensor<1x1xui32>, %arg152: tensor<1x1xui32>, %arg153: tensor<4xui32>, %arg154: tensor<4xui32>):
      %473 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %474 = "stablehlo.compare"(%arg146, %473) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%474) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg137: tensor<i64>, %arg138: tensor<i64>, %arg139: tensor<1x1xui32>, %arg140: tensor<1x1xui32>, %arg141: tensor<1x1xui32>, %arg142: tensor<1x1xui32>, %arg143: tensor<1x1xui32>, %arg144: tensor<4xui32>, %arg145: tensor<4xui32>):
      %470:8 = "func.call"(%arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145) <{callee = @None}> : (tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>)
      %471 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %472 = "stablehlo.add"(%arg137, %471) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%472, %470#0, %470#1, %470#2, %470#3, %470#4, %470#5, %470#6, %470#7) : (tensor<i64>, tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>)
    %468 = "stablehlo.concatenate"(%467#2, %467#3) <{dimension = 1 : i64}> : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x2xui32>
    %469 = "stablehlo.reshape"(%468) : (tensor<1x2xui32>) -> tensor<1x1x2xui32>
    "func.return"(%469) : (tensor<1x1x2xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg128: tensor<i64>, %arg129: tensor<1x1xui32>, %arg130: tensor<1x1xui32>, %arg131: tensor<1x1xui32>, %arg132: tensor<1x1xui32>, %arg133: tensor<1x1xui32>, %arg134: tensor<4xui32>, %arg135: tensor<4xui32>):
    %393 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %394 = "stablehlo.add"(%arg128, %393) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %395 = "stablehlo.slice"(%arg134) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %396 = "stablehlo.reshape"(%395) : (tensor<1xui32>) -> tensor<ui32>
    %397 = "stablehlo.add"(%arg129, %arg130) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %398 = "stablehlo.broadcast_in_dim"(%396) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1x1xui32>
    %399 = "stablehlo.shift_left"(%arg130, %398) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %400 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %401 = "stablehlo.subtract"(%400, %396) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %402 = "stablehlo.broadcast_in_dim"(%401) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1x1xui32>
    %403 = "stablehlo.shift_right_logical"(%arg130, %402) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %404 = "stablehlo.or"(%399, %403) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %405 = "stablehlo.xor"(%397, %404) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %406 = "stablehlo.slice"(%arg134) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %407 = "stablehlo.reshape"(%406) : (tensor<1xui32>) -> tensor<ui32>
    %408 = "stablehlo.add"(%397, %405) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %409 = "stablehlo.broadcast_in_dim"(%407) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1x1xui32>
    %410 = "stablehlo.shift_left"(%405, %409) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %411 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %412 = "stablehlo.subtract"(%411, %407) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %413 = "stablehlo.broadcast_in_dim"(%412) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1x1xui32>
    %414 = "stablehlo.shift_right_logical"(%405, %413) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %415 = "stablehlo.or"(%410, %414) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %416 = "stablehlo.xor"(%408, %415) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %417 = "stablehlo.slice"(%arg134) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %418 = "stablehlo.reshape"(%417) : (tensor<1xui32>) -> tensor<ui32>
    %419 = "stablehlo.add"(%408, %416) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %420 = "stablehlo.broadcast_in_dim"(%418) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1x1xui32>
    %421 = "stablehlo.shift_left"(%416, %420) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %422 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %423 = "stablehlo.subtract"(%422, %418) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %424 = "stablehlo.broadcast_in_dim"(%423) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1x1xui32>
    %425 = "stablehlo.shift_right_logical"(%416, %424) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %426 = "stablehlo.or"(%421, %425) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %427 = "stablehlo.xor"(%419, %426) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %428 = "stablehlo.slice"(%arg134) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %429 = "stablehlo.reshape"(%428) : (tensor<1xui32>) -> tensor<ui32>
    %430 = "stablehlo.add"(%419, %427) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %431 = "stablehlo.broadcast_in_dim"(%429) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1x1xui32>
    %432 = "stablehlo.shift_left"(%427, %431) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %433 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %434 = "stablehlo.subtract"(%433, %429) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %435 = "stablehlo.broadcast_in_dim"(%434) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1x1xui32>
    %436 = "stablehlo.shift_right_logical"(%427, %435) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %437 = "stablehlo.or"(%432, %436) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %438 = "stablehlo.xor"(%430, %437) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %439 = "stablehlo.add"(%430, %arg131) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %440 = "stablehlo.add"(%438, %arg132) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    %441 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %442 = "stablehlo.add"(%arg128, %441) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %443 = "stablehlo.convert"(%442) : (tensor<i64>) -> tensor<ui32>
    %444 = "stablehlo.broadcast_in_dim"(%443) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1x1xui32>
    %445 = "stablehlo.add"(%440, %444) : (tensor<1x1xui32>, tensor<1x1xui32>) -> tensor<1x1xui32>
    "func.return"(%394, %439, %445, %arg132, %arg133, %arg131, %arg135, %arg134) : (tensor<i64>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<1x1xui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<2xui32>, tensor<f64>) -> tensor<f64>, sym_name = "None_0", sym_visibility = "private"}> ({
  ^bb0(%arg106: tensor<2xui32>, %arg107: tensor<f64>):
    %318 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %319 = "stablehlo.compare"(%arg107, %318) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %320 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %321 = "stablehlo.add"(%arg107, %320) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %322 = "stablehlo.select"(%319, %arg107, %321) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %323 = "stablehlo.constant"() <{value = dense<0.33333333333333331> : tensor<f64>}> : () -> tensor<f64>
    %324 = "stablehlo.subtract"(%322, %323) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %325 = "stablehlo.sqrt"(%324) : (tensor<f64>) -> tensor<f64>
    %326 = "stablehlo.constant"() <{value = dense<0.33333333333333331> : tensor<f64>}> : () -> tensor<f64>
    %327 = "stablehlo.divide"(%326, %325) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %328 = "func.call"(%arg106) <{callee = @_threefry_split_1}> : (tensor<2xui32>) -> tensor<2x2xui32>
    %329 = "stablehlo.slice"(%328) <{limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %330 = "stablehlo.reshape"(%329) : (tensor<1x2xui32>) -> tensor<2xui32>
    %331 = "stablehlo.slice"(%328) <{limit_indices = array<i64: 2, 2>, start_indices = array<i64: 1, 0>, strides = array<i64: 1, 1>}> : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %332 = "stablehlo.reshape"(%331) : (tensor<1x2xui32>) -> tensor<2xui32>
    %333 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %334 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %335 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %336:6 = "stablehlo.while"(%324, %327, %330, %333, %334, %335) ({
    ^bb0(%arg122: tensor<f64>, %arg123: tensor<f64>, %arg124: tensor<2xui32>, %arg125: tensor<f64>, %arg126: tensor<f64>, %arg127: tensor<f64>):
      %376 = "stablehlo.multiply"(%arg125, %arg125) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %377 = "stablehlo.constant"() <{value = dense<3.310000e-02> : tensor<f64>}> : () -> tensor<f64>
      %378 = "stablehlo.multiply"(%377, %376) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %379 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %380 = "stablehlo.subtract"(%379, %378) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %381 = "stablehlo.compare"(%arg127, %380) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %382 = "stablehlo.log"(%arg127) : (tensor<f64>) -> tensor<f64>
      %383 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f64>}> : () -> tensor<f64>
      %384 = "stablehlo.multiply"(%arg125, %383) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %385 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %386 = "stablehlo.subtract"(%385, %arg126) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %387 = "stablehlo.log"(%arg126) : (tensor<f64>) -> tensor<f64>
      %388 = "stablehlo.add"(%386, %387) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %389 = "stablehlo.multiply"(%arg122, %388) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %390 = "stablehlo.add"(%384, %389) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %391 = "stablehlo.compare"(%382, %390) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %392 = "stablehlo.and"(%381, %391) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%392) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg108: tensor<f64>, %arg109: tensor<f64>, %arg110: tensor<2xui32>, %arg111: tensor<f64>, %arg112: tensor<f64>, %arg113: tensor<f64>):
      %349 = "func.call"(%arg110) <{callee = @_threefry_split_3}> : (tensor<2xui32>) -> tensor<3x2xui32>
      %350 = "stablehlo.slice"(%349) <{limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<3x2xui32>) -> tensor<1x2xui32>
      %351 = "stablehlo.reshape"(%350) : (tensor<1x2xui32>) -> tensor<2xui32>
      %352 = "stablehlo.slice"(%349) <{limit_indices = array<i64: 2, 2>, start_indices = array<i64: 1, 0>, strides = array<i64: 1, 1>}> : (tensor<3x2xui32>) -> tensor<1x2xui32>
      %353 = "stablehlo.reshape"(%352) : (tensor<1x2xui32>) -> tensor<2xui32>
      %354 = "stablehlo.slice"(%349) <{limit_indices = array<i64: 3, 2>, start_indices = array<i64: 2, 0>, strides = array<i64: 1, 1>}> : (tensor<3x2xui32>) -> tensor<1x2xui32>
      %355 = "stablehlo.reshape"(%354) : (tensor<1x2xui32>) -> tensor<2xui32>
      %356 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %357 = "stablehlo.constant"() <{value = dense<-1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %358:4 = "stablehlo.while"(%arg109, %353, %356, %357) ({
      ^bb0(%arg118: tensor<f64>, %arg119: tensor<2xui32>, %arg120: tensor<f64>, %arg121: tensor<f64>):
        %374 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
        %375 = "stablehlo.compare"(%arg121, %374) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
        "stablehlo.return"(%375) : (tensor<i1>) -> ()
      }, {
      ^bb0(%arg114: tensor<f64>, %arg115: tensor<2xui32>, %arg116: tensor<f64>, %arg117: tensor<f64>):
        %365 = "func.call"(%arg115) <{callee = @_threefry_split_1}> : (tensor<2xui32>) -> tensor<2x2xui32>
        %366 = "stablehlo.slice"(%365) <{limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<2x2xui32>) -> tensor<1x2xui32>
        %367 = "stablehlo.reshape"(%366) : (tensor<1x2xui32>) -> tensor<2xui32>
        %368 = "stablehlo.slice"(%365) <{limit_indices = array<i64: 2, 2>, start_indices = array<i64: 1, 0>, strides = array<i64: 1, 1>}> : (tensor<2x2xui32>) -> tensor<1x2xui32>
        %369 = "stablehlo.reshape"(%368) : (tensor<1x2xui32>) -> tensor<2xui32>
        %370 = "func.call"(%369) <{callee = @_normal}> : (tensor<2xui32>) -> tensor<f64>
        %371 = "stablehlo.multiply"(%370, %arg114) : (tensor<f64>, tensor<f64>) -> tensor<f64>
        %372 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
        %373 = "stablehlo.add"(%372, %371) : (tensor<f64>, tensor<f64>) -> tensor<f64>
        "stablehlo.return"(%arg114, %367, %370, %373) : (tensor<f64>, tensor<2xui32>, tensor<f64>, tensor<f64>) -> ()
      }) : (tensor<f64>, tensor<2xui32>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui32>, tensor<f64>, tensor<f64>)
      %359 = "stablehlo.multiply"(%358#2, %358#2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %360 = "stablehlo.multiply"(%358#3, %358#3) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %361 = "stablehlo.multiply"(%360, %358#3) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %362 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %363 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %364 = "func.call"(%355, %362, %363) <{callee = @_uniform_6}> : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%arg108, %arg109, %351, %359, %361, %364) : (tensor<f64>, tensor<f64>, tensor<2xui32>, tensor<f64>, tensor<f64>, tensor<f64>) -> ()
    }) : (tensor<f64>, tensor<f64>, tensor<2xui32>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui32>, tensor<f64>, tensor<f64>, tensor<f64>)
    %337 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %338 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %339 = "func.call"(%332, %337, %338) <{callee = @_uniform_6}> : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %340 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %341 = "stablehlo.subtract"(%340, %339) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %342 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %343 = "stablehlo.divide"(%342, %arg107) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %344 = "stablehlo.power"(%341, %343) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %345 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %346 = "stablehlo.select"(%319, %345, %344) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %347 = "stablehlo.multiply"(%324, %336#4) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %348 = "stablehlo.multiply"(%347, %346) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "func.return"(%348) : (tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<2x2xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_threefry_split_1", sym_visibility = "private"}> ({
  ^bb0(%arg87: tensor<2xui32>):
    %292 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xui32>
    %293 = "stablehlo.slice"(%arg87) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %294 = "stablehlo.reshape"(%293) : (tensor<1xui32>) -> tensor<ui32>
    %295 = "stablehlo.slice"(%arg87) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %296 = "stablehlo.reshape"(%295) : (tensor<1xui32>) -> tensor<ui32>
    %297 = "stablehlo.slice"(%292) <{limit_indices = array<i64: 2>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<2xui32>
    %298 = "stablehlo.slice"(%292) <{limit_indices = array<i64: 4>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<2xui32>
    %299 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %300 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %301 = "stablehlo.xor"(%294, %296) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %302 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %303 = "stablehlo.xor"(%301, %302) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %304 = "stablehlo.broadcast_in_dim"(%294) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %305 = "stablehlo.add"(%297, %304) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %306 = "stablehlo.broadcast_in_dim"(%296) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %307 = "stablehlo.add"(%298, %306) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %308 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %309 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %310:9 = "stablehlo.while"(%309, %308, %305, %307, %296, %303, %294, %299, %300) ({
    ^bb0(%arg97: tensor<i64>, %arg98: tensor<i64>, %arg99: tensor<2xui32>, %arg100: tensor<2xui32>, %arg101: tensor<ui32>, %arg102: tensor<ui32>, %arg103: tensor<ui32>, %arg104: tensor<4xui32>, %arg105: tensor<4xui32>):
      %316 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %317 = "stablehlo.compare"(%arg97, %316) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%317) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg88: tensor<i64>, %arg89: tensor<i64>, %arg90: tensor<2xui32>, %arg91: tensor<2xui32>, %arg92: tensor<ui32>, %arg93: tensor<ui32>, %arg94: tensor<ui32>, %arg95: tensor<4xui32>, %arg96: tensor<4xui32>):
      %313:8 = "func.call"(%arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96) <{callee = @None_2}> : (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %314 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %315 = "stablehlo.add"(%arg88, %314) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%315, %313#0, %313#1, %313#2, %313#3, %313#4, %313#5, %313#6, %313#7) : (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %311 = "stablehlo.concatenate"(%310#2, %310#3) <{dimension = 0 : i64}> : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %312 = "stablehlo.reshape"(%311) : (tensor<4xui32>) -> tensor<2x2xui32>
    "func.return"(%312) : (tensor<2x2xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None_2", sym_visibility = "private"}> ({
  ^bb0(%arg79: tensor<i64>, %arg80: tensor<2xui32>, %arg81: tensor<2xui32>, %arg82: tensor<ui32>, %arg83: tensor<ui32>, %arg84: tensor<ui32>, %arg85: tensor<4xui32>, %arg86: tensor<4xui32>):
    %237 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %238 = "stablehlo.add"(%arg79, %237) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %239 = "stablehlo.slice"(%arg85) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %240 = "stablehlo.reshape"(%239) : (tensor<1xui32>) -> tensor<ui32>
    %241 = "stablehlo.add"(%arg80, %arg81) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %242 = "stablehlo.broadcast_in_dim"(%240) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %243 = "stablehlo.shift_left"(%arg81, %242) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %244 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %245 = "stablehlo.subtract"(%244, %240) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %246 = "stablehlo.broadcast_in_dim"(%245) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %247 = "stablehlo.shift_right_logical"(%arg81, %246) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %248 = "stablehlo.or"(%243, %247) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %249 = "stablehlo.xor"(%241, %248) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %250 = "stablehlo.slice"(%arg85) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %251 = "stablehlo.reshape"(%250) : (tensor<1xui32>) -> tensor<ui32>
    %252 = "stablehlo.add"(%241, %249) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %253 = "stablehlo.broadcast_in_dim"(%251) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %254 = "stablehlo.shift_left"(%249, %253) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %255 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %256 = "stablehlo.subtract"(%255, %251) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %257 = "stablehlo.broadcast_in_dim"(%256) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %258 = "stablehlo.shift_right_logical"(%249, %257) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %259 = "stablehlo.or"(%254, %258) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %260 = "stablehlo.xor"(%252, %259) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %261 = "stablehlo.slice"(%arg85) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %262 = "stablehlo.reshape"(%261) : (tensor<1xui32>) -> tensor<ui32>
    %263 = "stablehlo.add"(%252, %260) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %264 = "stablehlo.broadcast_in_dim"(%262) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %265 = "stablehlo.shift_left"(%260, %264) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %266 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %267 = "stablehlo.subtract"(%266, %262) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %268 = "stablehlo.broadcast_in_dim"(%267) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %269 = "stablehlo.shift_right_logical"(%260, %268) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %270 = "stablehlo.or"(%265, %269) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %271 = "stablehlo.xor"(%263, %270) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %272 = "stablehlo.slice"(%arg85) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %273 = "stablehlo.reshape"(%272) : (tensor<1xui32>) -> tensor<ui32>
    %274 = "stablehlo.add"(%263, %271) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %275 = "stablehlo.broadcast_in_dim"(%273) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %276 = "stablehlo.shift_left"(%271, %275) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %277 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %278 = "stablehlo.subtract"(%277, %273) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %279 = "stablehlo.broadcast_in_dim"(%278) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %280 = "stablehlo.shift_right_logical"(%271, %279) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %281 = "stablehlo.or"(%276, %280) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %282 = "stablehlo.xor"(%274, %281) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %283 = "stablehlo.broadcast_in_dim"(%arg82) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %284 = "stablehlo.add"(%274, %283) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %285 = "stablehlo.broadcast_in_dim"(%arg83) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %286 = "stablehlo.add"(%282, %285) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %287 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %288 = "stablehlo.add"(%arg79, %287) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %289 = "stablehlo.convert"(%288) : (tensor<i64>) -> tensor<ui32>
    %290 = "stablehlo.broadcast_in_dim"(%289) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %291 = "stablehlo.add"(%286, %290) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    "func.return"(%238, %284, %291, %arg83, %arg84, %arg82, %arg86, %arg85) : (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<3x2xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_threefry_split_3", sym_visibility = "private"}> ({
  ^bb0(%arg60: tensor<2xui32>):
    %211 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<6xui32>
    %212 = "stablehlo.slice"(%arg60) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %213 = "stablehlo.reshape"(%212) : (tensor<1xui32>) -> tensor<ui32>
    %214 = "stablehlo.slice"(%arg60) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %215 = "stablehlo.reshape"(%214) : (tensor<1xui32>) -> tensor<ui32>
    %216 = "stablehlo.slice"(%211) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<6xui32>) -> tensor<3xui32>
    %217 = "stablehlo.slice"(%211) <{limit_indices = array<i64: 6>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<6xui32>) -> tensor<3xui32>
    %218 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %219 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %220 = "stablehlo.xor"(%213, %215) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %221 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %222 = "stablehlo.xor"(%220, %221) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %223 = "stablehlo.broadcast_in_dim"(%213) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %224 = "stablehlo.add"(%216, %223) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %225 = "stablehlo.broadcast_in_dim"(%215) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %226 = "stablehlo.add"(%217, %225) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %227 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %228 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %229:9 = "stablehlo.while"(%228, %227, %224, %226, %215, %222, %213, %218, %219) ({
    ^bb0(%arg70: tensor<i64>, %arg71: tensor<i64>, %arg72: tensor<3xui32>, %arg73: tensor<3xui32>, %arg74: tensor<ui32>, %arg75: tensor<ui32>, %arg76: tensor<ui32>, %arg77: tensor<4xui32>, %arg78: tensor<4xui32>):
      %235 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %236 = "stablehlo.compare"(%arg70, %235) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%236) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg61: tensor<i64>, %arg62: tensor<i64>, %arg63: tensor<3xui32>, %arg64: tensor<3xui32>, %arg65: tensor<ui32>, %arg66: tensor<ui32>, %arg67: tensor<ui32>, %arg68: tensor<4xui32>, %arg69: tensor<4xui32>):
      %232:8 = "func.call"(%arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69) <{callee = @None_4}> : (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %233 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %234 = "stablehlo.add"(%arg61, %233) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%234, %232#0, %232#1, %232#2, %232#3, %232#4, %232#5, %232#6, %232#7) : (tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %230 = "stablehlo.concatenate"(%229#2, %229#3) <{dimension = 0 : i64}> : (tensor<3xui32>, tensor<3xui32>) -> tensor<6xui32>
    %231 = "stablehlo.reshape"(%230) : (tensor<6xui32>) -> tensor<3x2xui32>
    "func.return"(%231) : (tensor<3x2xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None_4", sym_visibility = "private"}> ({
  ^bb0(%arg52: tensor<i64>, %arg53: tensor<3xui32>, %arg54: tensor<3xui32>, %arg55: tensor<ui32>, %arg56: tensor<ui32>, %arg57: tensor<ui32>, %arg58: tensor<4xui32>, %arg59: tensor<4xui32>):
    %156 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %157 = "stablehlo.add"(%arg52, %156) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %158 = "stablehlo.slice"(%arg58) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %159 = "stablehlo.reshape"(%158) : (tensor<1xui32>) -> tensor<ui32>
    %160 = "stablehlo.add"(%arg53, %arg54) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %161 = "stablehlo.broadcast_in_dim"(%159) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %162 = "stablehlo.shift_left"(%arg54, %161) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %163 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %164 = "stablehlo.subtract"(%163, %159) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %165 = "stablehlo.broadcast_in_dim"(%164) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %166 = "stablehlo.shift_right_logical"(%arg54, %165) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %167 = "stablehlo.or"(%162, %166) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %168 = "stablehlo.xor"(%160, %167) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %169 = "stablehlo.slice"(%arg58) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %170 = "stablehlo.reshape"(%169) : (tensor<1xui32>) -> tensor<ui32>
    %171 = "stablehlo.add"(%160, %168) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %172 = "stablehlo.broadcast_in_dim"(%170) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %173 = "stablehlo.shift_left"(%168, %172) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %174 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %175 = "stablehlo.subtract"(%174, %170) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %176 = "stablehlo.broadcast_in_dim"(%175) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %177 = "stablehlo.shift_right_logical"(%168, %176) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %178 = "stablehlo.or"(%173, %177) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %179 = "stablehlo.xor"(%171, %178) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %180 = "stablehlo.slice"(%arg58) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %181 = "stablehlo.reshape"(%180) : (tensor<1xui32>) -> tensor<ui32>
    %182 = "stablehlo.add"(%171, %179) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %183 = "stablehlo.broadcast_in_dim"(%181) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %184 = "stablehlo.shift_left"(%179, %183) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %185 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %186 = "stablehlo.subtract"(%185, %181) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %187 = "stablehlo.broadcast_in_dim"(%186) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %188 = "stablehlo.shift_right_logical"(%179, %187) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %189 = "stablehlo.or"(%184, %188) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %190 = "stablehlo.xor"(%182, %189) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %191 = "stablehlo.slice"(%arg58) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %192 = "stablehlo.reshape"(%191) : (tensor<1xui32>) -> tensor<ui32>
    %193 = "stablehlo.add"(%182, %190) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %194 = "stablehlo.broadcast_in_dim"(%192) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %195 = "stablehlo.shift_left"(%190, %194) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %196 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %197 = "stablehlo.subtract"(%196, %192) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %198 = "stablehlo.broadcast_in_dim"(%197) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %199 = "stablehlo.shift_right_logical"(%190, %198) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %200 = "stablehlo.or"(%195, %199) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %201 = "stablehlo.xor"(%193, %200) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %202 = "stablehlo.broadcast_in_dim"(%arg55) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %203 = "stablehlo.add"(%193, %202) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %204 = "stablehlo.broadcast_in_dim"(%arg56) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %205 = "stablehlo.add"(%201, %204) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %206 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %207 = "stablehlo.add"(%arg52, %206) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %208 = "stablehlo.convert"(%207) : (tensor<i64>) -> tensor<ui32>
    %209 = "stablehlo.broadcast_in_dim"(%208) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %210 = "stablehlo.add"(%205, %209) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    "func.return"(%157, %203, %210, %arg56, %arg57, %arg55, %arg59, %arg58) : (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<f64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_normal", sym_visibility = "private"}> ({
  ^bb0(%arg51: tensor<2xui32>):
    %155 = "func.call"(%arg51) <{callee = @_normal_real}> : (tensor<2xui32>) -> tensor<f64>
    "func.return"(%155) : (tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<f64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_normal_real", sym_visibility = "private"}> ({
  ^bb0(%arg50: tensor<2xui32>):
    %149 = "stablehlo.constant"() <{value = dense<-0.99999999999999988> : tensor<f64>}> : () -> tensor<f64>
    %150 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %151 = "func.call"(%arg50, %149, %150) <{callee = @_uniform}> : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %152 = "chlo.erf_inv"(%151) : (tensor<f64>) -> tensor<f64>
    %153 = "stablehlo.constant"() <{value = dense<1.4142135623730951> : tensor<f64>}> : () -> tensor<f64>
    %154 = "stablehlo.multiply"(%153, %152) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "func.return"(%154) : (tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<f64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_uniform", sym_visibility = "private"}> ({
  ^bb0(%arg29: tensor<2xui32>, %arg30: tensor<f64>, %arg31: tensor<f64>):
    %103 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xui32>
    %104 = "stablehlo.slice"(%arg29) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %105 = "stablehlo.reshape"(%104) : (tensor<1xui32>) -> tensor<ui32>
    %106 = "stablehlo.slice"(%arg29) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %107 = "stablehlo.reshape"(%106) : (tensor<1xui32>) -> tensor<ui32>
    %108 = "stablehlo.slice"(%103) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %109 = "stablehlo.slice"(%103) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %110 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %111 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %112 = "stablehlo.xor"(%105, %107) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %113 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %114 = "stablehlo.xor"(%112, %113) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %115 = "stablehlo.broadcast_in_dim"(%105) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %116 = "stablehlo.add"(%108, %115) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %117 = "stablehlo.broadcast_in_dim"(%107) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %118 = "stablehlo.add"(%109, %117) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %119 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %120 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %121:9 = "stablehlo.while"(%120, %119, %116, %118, %107, %114, %105, %110, %111) ({
    ^bb0(%arg41: tensor<i64>, %arg42: tensor<i64>, %arg43: tensor<1xui32>, %arg44: tensor<1xui32>, %arg45: tensor<ui32>, %arg46: tensor<ui32>, %arg47: tensor<ui32>, %arg48: tensor<4xui32>, %arg49: tensor<4xui32>):
      %147 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %148 = "stablehlo.compare"(%arg41, %147) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%148) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg32: tensor<i64>, %arg33: tensor<i64>, %arg34: tensor<1xui32>, %arg35: tensor<1xui32>, %arg36: tensor<ui32>, %arg37: tensor<ui32>, %arg38: tensor<ui32>, %arg39: tensor<4xui32>, %arg40: tensor<4xui32>):
      %144:8 = "func.call"(%arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40) <{callee = @None_5}> : (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %145 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %146 = "stablehlo.add"(%arg32, %145) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%146, %144#0, %144#1, %144#2, %144#3, %144#4, %144#5, %144#6, %144#7) : (tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %122 = "stablehlo.concatenate"(%121#2, %121#3) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %123 = "stablehlo.slice"(%122) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %124 = "stablehlo.slice"(%122) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %125 = "stablehlo.convert"(%123) : (tensor<1xui32>) -> tensor<1xui64>
    %126 = "stablehlo.convert"(%124) : (tensor<1xui32>) -> tensor<1xui64>
    %127 = "stablehlo.constant"() <{value = dense<32> : tensor<ui64>}> : () -> tensor<ui64>
    %128 = "stablehlo.broadcast_in_dim"(%127) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<1xui64>
    %129 = "stablehlo.shift_left"(%125, %128) : (tensor<1xui64>, tensor<1xui64>) -> tensor<1xui64>
    %130 = "stablehlo.or"(%129, %126) : (tensor<1xui64>, tensor<1xui64>) -> tensor<1xui64>
    %131 = "stablehlo.reshape"(%130) : (tensor<1xui64>) -> tensor<ui64>
    %132 = "stablehlo.constant"() <{value = dense<12> : tensor<ui64>}> : () -> tensor<ui64>
    %133 = "stablehlo.shift_right_logical"(%131, %132) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
    %134 = "stablehlo.constant"() <{value = dense<4607182418800017408> : tensor<ui64>}> : () -> tensor<ui64>
    %135 = "stablehlo.or"(%133, %134) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
    %136 = "stablehlo.bitcast_convert"(%135) : (tensor<ui64>) -> tensor<f64>
    %137 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %138 = "stablehlo.subtract"(%136, %137) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %139 = "stablehlo.subtract"(%arg31, %arg30) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %140 = "stablehlo.multiply"(%138, %139) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %141 = "stablehlo.add"(%140, %arg30) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %142 = "stablehlo.reshape"(%141) : (tensor<f64>) -> tensor<f64>
    %143 = "stablehlo.maximum"(%arg30, %142) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "func.return"(%143) : (tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None_5", sym_visibility = "private"}> ({
  ^bb0(%arg21: tensor<i64>, %arg22: tensor<1xui32>, %arg23: tensor<1xui32>, %arg24: tensor<ui32>, %arg25: tensor<ui32>, %arg26: tensor<ui32>, %arg27: tensor<4xui32>, %arg28: tensor<4xui32>):
    %48 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %49 = "stablehlo.add"(%arg21, %48) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %50 = "stablehlo.slice"(%arg27) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %51 = "stablehlo.reshape"(%50) : (tensor<1xui32>) -> tensor<ui32>
    %52 = "stablehlo.add"(%arg22, %arg23) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %53 = "stablehlo.broadcast_in_dim"(%51) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %54 = "stablehlo.shift_left"(%arg23, %53) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %55 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %56 = "stablehlo.subtract"(%55, %51) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %57 = "stablehlo.broadcast_in_dim"(%56) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %58 = "stablehlo.shift_right_logical"(%arg23, %57) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %59 = "stablehlo.or"(%54, %58) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %60 = "stablehlo.xor"(%52, %59) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %61 = "stablehlo.slice"(%arg27) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %62 = "stablehlo.reshape"(%61) : (tensor<1xui32>) -> tensor<ui32>
    %63 = "stablehlo.add"(%52, %60) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %64 = "stablehlo.broadcast_in_dim"(%62) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %65 = "stablehlo.shift_left"(%60, %64) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %66 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %67 = "stablehlo.subtract"(%66, %62) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %68 = "stablehlo.broadcast_in_dim"(%67) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %69 = "stablehlo.shift_right_logical"(%60, %68) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %70 = "stablehlo.or"(%65, %69) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %71 = "stablehlo.xor"(%63, %70) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %72 = "stablehlo.slice"(%arg27) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %73 = "stablehlo.reshape"(%72) : (tensor<1xui32>) -> tensor<ui32>
    %74 = "stablehlo.add"(%63, %71) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %75 = "stablehlo.broadcast_in_dim"(%73) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %76 = "stablehlo.shift_left"(%71, %75) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %77 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %78 = "stablehlo.subtract"(%77, %73) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %79 = "stablehlo.broadcast_in_dim"(%78) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %80 = "stablehlo.shift_right_logical"(%71, %79) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %81 = "stablehlo.or"(%76, %80) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %82 = "stablehlo.xor"(%74, %81) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %83 = "stablehlo.slice"(%arg27) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %84 = "stablehlo.reshape"(%83) : (tensor<1xui32>) -> tensor<ui32>
    %85 = "stablehlo.add"(%74, %82) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %86 = "stablehlo.broadcast_in_dim"(%84) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %87 = "stablehlo.shift_left"(%82, %86) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %88 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %89 = "stablehlo.subtract"(%88, %84) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %90 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %91 = "stablehlo.shift_right_logical"(%82, %90) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %92 = "stablehlo.or"(%87, %91) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %93 = "stablehlo.xor"(%85, %92) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %94 = "stablehlo.broadcast_in_dim"(%arg24) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %95 = "stablehlo.add"(%85, %94) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %96 = "stablehlo.broadcast_in_dim"(%arg25) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %97 = "stablehlo.add"(%93, %96) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %98 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %99 = "stablehlo.add"(%arg21, %98) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %100 = "stablehlo.convert"(%99) : (tensor<i64>) -> tensor<ui32>
    %101 = "stablehlo.broadcast_in_dim"(%100) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %102 = "stablehlo.add"(%97, %101) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    "func.return"(%49, %95, %102, %arg25, %arg26, %arg24, %arg28, %arg27) : (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<f64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_uniform_6", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<2xui32>, %arg1: tensor<f64>, %arg2: tensor<f64>):
    %0 = "stablehlo.convert"(%arg1) : (tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.convert"(%arg2) : (tensor<f64>) -> tensor<f64>
    %2 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2xui32>
    %3 = "stablehlo.slice"(%arg0) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %4 = "stablehlo.reshape"(%3) : (tensor<1xui32>) -> tensor<ui32>
    %5 = "stablehlo.slice"(%arg0) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %6 = "stablehlo.reshape"(%5) : (tensor<1xui32>) -> tensor<ui32>
    %7 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %8 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %9 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %10 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %11 = "stablehlo.xor"(%4, %6) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %12 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %13 = "stablehlo.xor"(%11, %12) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %14 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %15 = "stablehlo.add"(%7, %14) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %16 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %17 = "stablehlo.add"(%8, %16) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %18 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %19 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %20:9 = "stablehlo.while"(%19, %18, %15, %17, %6, %13, %4, %9, %10) ({
    ^bb0(%arg12: tensor<i64>, %arg13: tensor<i64>, %arg14: tensor<1xui32>, %arg15: tensor<1xui32>, %arg16: tensor<ui32>, %arg17: tensor<ui32>, %arg18: tensor<ui32>, %arg19: tensor<4xui32>, %arg20: tensor<4xui32>):
      %46 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %47 = "stablehlo.compare"(%arg12, %46) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%47) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: tensor<1xui32>, %arg6: tensor<1xui32>, %arg7: tensor<ui32>, %arg8: tensor<ui32>, %arg9: tensor<ui32>, %arg10: tensor<4xui32>, %arg11: tensor<4xui32>):
      %43:8 = "func.call"(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11) <{callee = @None_5}> : (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %44 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %45 = "stablehlo.add"(%arg3, %44) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%45, %43#0, %43#1, %43#2, %43#3, %43#4, %43#5, %43#6, %43#7) : (tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %21 = "stablehlo.concatenate"(%20#2, %20#3) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %22 = "stablehlo.slice"(%21) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %23 = "stablehlo.slice"(%21) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %24 = "stablehlo.convert"(%22) : (tensor<1xui32>) -> tensor<1xui64>
    %25 = "stablehlo.convert"(%23) : (tensor<1xui32>) -> tensor<1xui64>
    %26 = "stablehlo.constant"() <{value = dense<32> : tensor<ui64>}> : () -> tensor<ui64>
    %27 = "stablehlo.broadcast_in_dim"(%26) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<1xui64>
    %28 = "stablehlo.shift_left"(%24, %27) : (tensor<1xui64>, tensor<1xui64>) -> tensor<1xui64>
    %29 = "stablehlo.or"(%28, %25) : (tensor<1xui64>, tensor<1xui64>) -> tensor<1xui64>
    %30 = "stablehlo.reshape"(%29) : (tensor<1xui64>) -> tensor<ui64>
    %31 = "stablehlo.constant"() <{value = dense<12> : tensor<ui64>}> : () -> tensor<ui64>
    %32 = "stablehlo.shift_right_logical"(%30, %31) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
    %33 = "stablehlo.constant"() <{value = dense<4607182418800017408> : tensor<ui64>}> : () -> tensor<ui64>
    %34 = "stablehlo.or"(%32, %33) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
    %35 = "stablehlo.bitcast_convert"(%34) : (tensor<ui64>) -> tensor<f64>
    %36 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %37 = "stablehlo.subtract"(%35, %36) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %38 = "stablehlo.subtract"(%1, %0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %39 = "stablehlo.multiply"(%37, %38) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %40 = "stablehlo.add"(%39, %0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %41 = "stablehlo.reshape"(%40) : (tensor<f64>) -> tensor<f64>
    %42 = "stablehlo.maximum"(%0, %41) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "func.return"(%42) : (tensor<f64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

